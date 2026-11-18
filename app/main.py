"""FastAPI 磁盘风险推理服务。

POST /predict 提交 SMART 测值，返回 0~1 风险分数与推理批次号；
成功的预测同时登记可追溯回执，运维凭回执提交实际巡检结论，
再按回执查询误报 / 漏报统计。
Swagger UI: /docs ，OpenAPI: /openapi.json 。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .batcher import BatchInferenceError, Batcher, QueueFullError
from .config import settings
from .model import LogisticRiskModel
from .receipts import (
    HIGH_RISK_THRESHOLD,
    ReceiptNotFoundError,
    ReceiptStore,
    ReceiptStoreUnavailable,
)

logger = logging.getLogger("disk_risk.api")


class PredictRequest(BaseModel):
    disk_no: int = Field(..., ge=0, le=65535, description="磁盘号")
    temperature: float = Field(..., ge=-40.0, le=150.0, description="温度(℃)")
    reallocated_sectors: int = Field(..., ge=0, description="重映射扇区数")
    pending_sectors: int = Field(..., ge=0, description="待映射扇区数")
    deadline_ms: int = Field(
        ...,
        ge=1,
        le=60000,
        description="从接入起算的独立总期限(毫秒)，超时返回 504",
    )


class PredictResponse(BaseModel):
    disk_no: int
    risk_score: float = Field(..., ge=0.0, le=1.0, description="0~1 风险分数")
    batch_id: int = Field(..., description="推理批次号（每次矩阵推理递增）")
    receipt_id: str | None = Field(
        ...,
        description=(
            "可追溯回执号；仅请求期限内成功（200）的预测发放。"
            "回执库故障时为 null，此类分数不可提交巡检反馈"
        ),
    )


class FeedbackRequest(BaseModel):
    actual_abnormal: bool = Field(
        ..., description="巡检实际结论：该盘是否异常（true=实际异常）"
    )


class FeedbackResponse(BaseModel):
    receipt_id: str
    actual_abnormal: bool = Field(..., description="当前生效的巡检结论")
    revision: int = Field(..., description="结论修订号；更正一次递增 1")
    corrected: bool = Field(..., description="本次提交是否更正了既有结论")
    created_at: str
    updated_at: str


class ReceiptView(BaseModel):
    receipt_id: str
    disk_no: int
    risk_score: float
    batch_id: int
    created_at: str
    feedback: FeedbackResponse | None = Field(
        None, description="当前生效的巡检反馈；未反馈时为 null"
    )


class ReportResponse(BaseModel):
    high_risk_threshold: float = Field(
        ..., description="高风险界线：risk_score >= 该值视为高风险"
    )
    feedback_samples: int = Field(..., description="已反馈样本数（按回执计样）")
    false_positives: int = Field(
        ..., description="误报数：预测高风险(>=0.5)但实际正常"
    )
    false_negatives: int = Field(..., description="漏报数：预测低风险(<0.5)但实际异常")


_batcher: Batcher | None = None
_model: LogisticRiskModel | None = None
_receipts: ReceiptStore | None = None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    global _batcher, _model, _receipts
    _model = LogisticRiskModel(
        latency_s=settings.model_latency_s,
        fail_batch_no=settings.fail_batch_no,
    )
    _batcher = Batcher(_model)

    # 回执库初始化失败不阻断推理：线上仍按原期限给分，只是暂不发回执；
    # 后续请求会重试初始化，恢复后自动继续发回执。
    _receipts = ReceiptStore(
        settings.receipt_db_path,
        busy_timeout_ms=settings.receipt_db_busy_timeout_ms,
        write_timeout_s=settings.receipt_write_timeout_s,
    )
    try:
        await asyncio.to_thread(_receipts.initialize)
    except ReceiptStoreUnavailable as exc:
        logger.warning("receipt store unavailable at startup: %s", exc)

    await _batcher.start()
    try:
        yield
    finally:
        await _batcher.stop()
        if _receipts is not None:
            await asyncio.to_thread(_receipts.close)


app = FastAPI(
    title="磁盘 SMART 风险推理 API",
    version="1.1.0",
    description=(
        "固定权重 NumPy 逻辑回归 + 异步合批（单批≤8，窗口≤40ms，排队≤32）。"
        "成功预测发放 SQLite 回执，运维按回执提交巡检结论，"
        "查询以 risk_score≥0.5 为高风险界线的误报 / 漏报 / 已反馈样本数。"
    ),
    lifespan=lifespan,
)


@app.exception_handler(QueueFullError)
async def _queue_full_handler(_request: Request, exc: QueueFullError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
        headers={"Retry-After": "0"},
    )


@app.exception_handler(TimeoutError)
async def _timeout_handler(_request: Request, exc: TimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.exception_handler(BatchInferenceError)
async def _batch_fail_handler(
    _request: Request, exc: BatchInferenceError
) -> JSONResponse:
    # 模型处理失败：该批报错（500），后续批仍可正常工作；该批不发回执
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.exception_handler(ReceiptNotFoundError)
async def _receipt_not_found_handler(
    _request: Request, exc: ReceiptNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ReceiptStoreUnavailable)
async def _receipt_store_handler(
    _request: Request, exc: ReceiptStoreUnavailable
) -> JSONResponse:
    # 回执库不可用：反馈读 / 写无法可靠落盘，明确拒绝而非假装成功
    return JSONResponse(
        status_code=503,
        content={"detail": f"receipt store unavailable: {exc}"},
        headers={"Retry-After": "1"},
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "queue_depth": _batcher.queue_depth if _batcher else -1,
        "receipt_store_ready": bool(_receipts and _receipts.is_ready),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="提交 SMART 测值并返回风险分数与可追溯回执",
)
async def predict(req: PredictRequest, request: Request) -> PredictResponse:
    assert _batcher is not None
    features = (
        float(req.temperature),
        float(req.reallocated_sectors),
        float(req.pending_sectors),
        float(req.disk_no),
    )
    deadline_s = req.deadline_ms / 1000.0

    # 入队（满员 -> 503）
    entry = await _batcher.enqueue(features, deadline_s)
    result_task = asyncio.create_task(_batcher.await_result(entry))

    try:
        # Starlette 的 is_disconnected() 是非阻塞一次性探询，需轮询；
        # 轮询间隔远小于组批窗口，断连后可快速腾出名额
        while not result_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(result_task), timeout=0.02)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    result_task.cancel()
                    await _batcher.discard(entry)
                    return JSONResponse(  # type: ignore[return-value]
                        status_code=499,
                        content={"detail": "client disconnected"},
                    )
        # 期限超时(504) / 整批模型失败(500) 由异常处理器统一应答
        score, batch_id = result_task.result()
    finally:
        if not result_task.done():
            result_task.cancel()
        elif not result_task.cancelled():
            result_task.exception()  # 回收异常，避免任务告警

    # 只有“请求期限内成功”的分数才登记回执；回执库故障不影响本次给分，
    # 但不会发出不可填报的假回执。
    receipt_id = await _register_receipt(req.disk_no, score, batch_id)

    return PredictResponse(
        disk_no=req.disk_no,
        risk_score=score,
        batch_id=batch_id,
        receipt_id=receipt_id,
    )


async def _register_receipt(
    disk_no: int, score: float, batch_id: int
) -> str | None:
    assert _receipts is not None
    try:
        receipt = await asyncio.to_thread(
            _receipts.register, disk_no=disk_no, risk_score=score, batch_id=batch_id
        )
        return receipt.receipt_id
    except ReceiptStoreUnavailable as exc:
        logger.warning(
            "receipt registration failed; score returned without receipt: %s", exc
        )
        return None


def _feedback_response(feedback) -> FeedbackResponse:
    return FeedbackResponse(
        receipt_id=feedback.receipt_id,
        actual_abnormal=feedback.actual_abnormal,
        revision=feedback.revision,
        corrected=False,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


@app.get(
    "/receipts/{receipt_id}",
    response_model=ReceiptView,
    summary="按回执号查询预测记录与当前巡检结论",
    responses={404: {"description": "回执不存在"}},
)
async def get_receipt(receipt_id: str) -> ReceiptView:
    assert _receipts is not None
    record = await asyncio.to_thread(_receipts.get_receipt, receipt_id)
    return ReceiptView(
        receipt_id=record.receipt.receipt_id,
        disk_no=record.receipt.disk_no,
        risk_score=record.receipt.risk_score,
        batch_id=record.receipt.batch_id,
        created_at=record.receipt.created_at,
        feedback=(
            _feedback_response(record.feedback) if record.feedback else None
        ),
    )


@app.post(
    "/receipts/{receipt_id}/feedback",
    response_model=FeedbackResponse,
    summary="提交/更正巡检实际结论",
    responses={
        404: {"description": "回执不存在"},
        503: {"description": "回执库不可用"},
    },
)
async def submit_feedback(
    receipt_id: str, req: FeedbackRequest
) -> FeedbackResponse:
    """同一回执重复提交相同结论不重复计数；不同结论覆盖旧值，报告只认最新结论。"""
    assert _receipts is not None
    feedback, corrected = await asyncio.to_thread(
        _receipts.submit_feedback, receipt_id, req.actual_abnormal
    )
    return FeedbackResponse(
        receipt_id=feedback.receipt_id,
        actual_abnormal=feedback.actual_abnormal,
        revision=feedback.revision,
        corrected=corrected,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
    )


@app.get(
    "/feedback/report",
    response_model=ReportResponse,
    summary="查询误报数、漏报数与已反馈样本数（高风险界线 0.5）",
)
async def feedback_report() -> ReportResponse:
    """按每个回执的最新结论统计；同一盘的不同预测分别计样。

    查询在单个 SQLite 读事务内完成，与并发更正不会混用新旧记录。
    """
    assert _receipts is not None
    report = await asyncio.to_thread(_receipts.report)
    return ReportResponse(
        high_risk_threshold=report.high_risk_threshold,
        feedback_samples=report.feedback_samples,
        false_positives=report.false_positives,
        false_negatives=report.false_negatives,
    )


@app.get("/stats")
async def stats() -> dict:
    """运行期指标，便于自测核对合批行为。"""
    assert _batcher is not None
    receipt_store = {"ready": bool(_receipts and _receipts.is_ready), "last_error": None}
    if _receipts is not None:
        receipt_store["last_error"] = _receipts.last_error
        if _receipts.is_ready:
            try:
                receipt_store.update(
                    await asyncio.to_thread(_receipts.counts)
                )
            except ReceiptStoreUnavailable as exc:
                receipt_store["last_error"] = str(exc)
    return {
        "queue_depth": _batcher.queue_depth,
        "total_batches": _batcher.total_batches,
        "total_inferences": _batcher.total_inferences,
        "failed_batches": _batcher.failed_batches,
        "receipt_store": receipt_store,
        "config": {
            "max_batch_size": _batcher.max_batch_size,
            "batch_window_ms": settings.batch_window_ms,
            "max_queue": _batcher.max_queue,
            "model_latency_s": settings.model_latency_s,
            "fail_batch_no": settings.fail_batch_no,
            "high_risk_threshold": HIGH_RISK_THRESHOLD,
        },
    }


@app.post("/debug/arm-failure", tags=["debug"])
async def arm_failure() -> dict:
    """【测试/演示】武装一次模型故障：下一次矩阵推理必定整批失败，仅一次。

    用于确定性验证“该批报错、后续批仍可工作”，无需按批次号撞时机。
    生产环境可通过反向代理屏蔽该路由。
    """
    assert _model is not None
    _model.arm_failure_once()
    return {"armed": True, "scope": "next inference, one shot"}
