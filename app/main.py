"""FastAPI 磁盘风险推理服务。

POST /predict 提交 SMART 测值，返回 0~1 风险分数与推理批次号。
Swagger UI: /docs ，OpenAPI: /openapi.json 。
"""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .batcher import BatchInferenceError, Batcher, QueueFullError
from .config import settings
from .model import LogisticRiskModel


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


_batcher: Batcher | None = None
_model: LogisticRiskModel | None = None


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    global _batcher, _model
    _model = LogisticRiskModel(
        latency_s=settings.model_latency_s,
        fail_batch_no=settings.fail_batch_no,
    )
    _batcher = Batcher(_model)
    await _batcher.start()
    try:
        yield
    finally:
        await _batcher.stop()


app = FastAPI(
    title="磁盘 SMART 风险推理 API",
    version="1.0.0",
    description="固定权重 NumPy 逻辑回归 + 异步合批（单批≤8，窗口≤40ms，排队≤32）",
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
    # 模型处理失败：该批报错（500），后续批仍可正常工作
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "queue_depth": _batcher.queue_depth if _batcher else -1}


@app.post("/predict", response_model=PredictResponse)
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

    return PredictResponse(
        disk_no=req.disk_no, risk_score=score, batch_id=batch_id
    )


@app.get("/stats")
async def stats() -> dict:
    """运行期指标，便于自测核对合批行为。"""
    assert _batcher is not None
    return {
        "queue_depth": _batcher.queue_depth,
        "total_batches": _batcher.total_batches,
        "total_inferences": _batcher.total_inferences,
        "failed_batches": _batcher.failed_batches,
        "config": {
            "max_batch_size": _batcher.max_batch_size,
            "batch_window_ms": settings.batch_window_ms,
            "max_queue": _batcher.max_queue,
            "model_latency_s": settings.model_latency_s,
            "fail_batch_no": settings.fail_batch_no,
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
