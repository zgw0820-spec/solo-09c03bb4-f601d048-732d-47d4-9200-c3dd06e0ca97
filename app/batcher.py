"""异步组批引擎。

并发的推理请求在此合批：
* 单批最多 max_batch_size 个（默认 8）；
* 组批窗口不超过 batch_window_s（默认 40ms）；
* 排队名额 max_queue 个（默认 32），满员直接拒绝（503）；
* 每个请求带“从接入起算”的独立总期限 deadline，到期只影响自己，
  不拖累同批其他请求；排队超时或客户端断连会腾出名额；
* 一批内模型处理失败只使该批报错，不影响后续批次。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from .config import settings
from .model import LogisticRiskModel

logger = logging.getLogger("disk_risk.batcher")


class QueueFullError(RuntimeError):
    """排队名额已满。"""


class BatchInferenceError(RuntimeError):
    """该批模型处理失败。"""


@dataclass
class _Entry:
    features: tuple[float, float, float, float]
    deadline: float                      # loop.time() 基准的绝对期限
    future: "asyncio.Future[tuple[float, int]]"
    enqueued_at: float


class Batcher:
    def __init__(
        self,
        model: LogisticRiskModel,
        max_batch_size: int | None = None,
        batch_window_s: float | None = None,
        max_queue: int | None = None,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size or settings.max_batch_size
        self.batch_window_s = (
            settings.batch_window_s if batch_window_s is None else batch_window_s
        )
        self.max_queue = max_queue or settings.max_queue

        self._cond = asyncio.Condition()
        self._pending: list[_Entry] = []
        self._task: asyncio.Task | None = None
        self._batch_seq = 0

        # 可观测指标
        self.total_batches = 0
        self.total_inferences = 0
        self.failed_batches = 0

    # ---------- 生命周期 ----------
    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="batcher")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # 释放仍在排队的请求
        async with self._cond:
            for entry in self._pending:
                if not entry.future.done():
                    entry.future.set_exception(
                        BatchInferenceError("service shutting down")
                    )
            self._pending.clear()
            self._cond.notify_all()

    # ---------- 请求接入 ----------
    @property
    def queue_depth(self) -> int:
        return len(self._pending)

    async def enqueue(
        self, features: tuple[float, float, float, float], deadline_s: float
    ) -> "_Entry":
        """入队一个推理请求，返回排队条目。队列满抛 QueueFullError。"""
        loop = asyncio.get_running_loop()
        entry = _Entry(
            features=features,
            deadline=loop.time() + max(0.0, deadline_s),
            future=loop.create_future(),
            enqueued_at=loop.time(),
        )
        async with self._cond:
            if len(self._pending) >= self.max_queue:
                raise QueueFullError(
                    f"queue full ({self.max_queue} slots), request rejected"
                )
            self._pending.append(entry)
            self._cond.notify_all()
        return entry

    async def await_result(self, entry: "_Entry") -> tuple[float, int]:
        """等待结果，受该请求自身的独立总期限约束。

        期限先到：排队中则移除腾出名额；已成批则丢弃迟到结果，绝不报成功。
        """
        loop = asyncio.get_running_loop()
        timeout = max(0.0, entry.deadline - loop.time())
        try:
            await asyncio.wait_for(asyncio.shield(entry.future), timeout=timeout)
        except asyncio.TimeoutError:
            await self.discard(entry)
            raise TimeoutError("request deadline exceeded")
        return entry.future.result()

    async def submit(
        self, features: tuple[float, float, float, float], deadline_s: float
    ) -> tuple[float, int]:
        """入队并等待结果的便捷封装。"""
        entry = await self.enqueue(features, deadline_s)
        return await self.await_result(entry)

    async def discard(self, entry: _Entry) -> None:
        """请求放弃（排队超时 / 客户端断连）：仍在排队则移除并腾出名额。"""
        async with self._cond:
            try:
                self._pending.remove(entry)
            except ValueError:
                pass  # 已经成批，名额本就不占队列
            self._cond.notify_all()

    # ---------- 组批主循环 ----------
    async def _run(self) -> None:
        # 串行推理：同一时刻只有一批矩阵推理在执行，排队名额才是真实容量
        while True:
            try:
                batch, batch_no = await self._collect_batch()
            except asyncio.CancelledError:
                raise
            except Exception:
                # 组批逻辑自身异常不能静默吞掉
                logger.exception("batcher collect loop error")
                await asyncio.sleep(0.05)
                continue
            if not batch:
                continue
            try:
                await self._process_batch(batch, batch_no)
            except Exception:
                # 兜底：处理循环本身异常也不能拖垮后续批次
                logger.exception("batcher process loop error")

    async def _collect_batch(self) -> tuple[list[_Entry], int]:
        loop = asyncio.get_running_loop()
        async with self._cond:
            while not self._pending:
                await self._cond.wait()

            # 窗口起点 = 队首（最早）请求的入队时刻
            window_start = self._pending[0].enqueued_at
            while True:
                now = loop.time()
                depth = len(self._pending)

                if depth >= self.max_batch_size:
                    break  # 满批，立即出发
                if now - window_start >= self.batch_window_s:
                    break  # 窗口到期，立即出发

                # 睡到：窗口结束 或 新请求到达（被 notify 唤醒后重算）
                # 注意：Python 3.11 的 Condition.wait 不支持 timeout，用 wait_for 包裹
                remaining = max(0.0, self.batch_window_s - (now - window_start))
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    pass  # 窗口自然到期，回循环重判并出批

            self._batch_seq += 1
            batch_no = self._batch_seq
            take = min(self.max_batch_size, len(self._pending))
            batch = self._pending[:take]
            del self._pending[:take]
            return batch, batch_no

    async def _process_batch(self, batch: list[_Entry], batch_no: int) -> None:
        self.total_batches += 1
        loop = asyncio.get_running_loop()
        try:
            matrix = np.asarray([e.features for e in batch], dtype=np.float64)
            # 模拟推理延迟（阻塞 sleep 放线程池，不卡住事件循环）
            if self.model.latency_s > 0:
                await asyncio.to_thread(time.sleep, self.model.latency_s)
            scores = await asyncio.to_thread(self.model.predict, matrix)
            self.total_inferences += len(batch)
        except Exception as exc:  # noqa: BLE001 - 模型侧任何异常都按整批失败处理
            # 整批失败：只影响本批，主循环与后续批不受影响
            self.failed_batches += 1
            err = BatchInferenceError(f"batch {batch_no} model failure: {exc}")
            for entry in batch:
                if not entry.future.done():
                    entry.future.set_exception(err)
            return

        now = loop.time()
        for entry, score in zip(batch, scores):
            if entry.future.done():
                continue  # 已超时/断连，结果直接丢弃
            # 成批后单个请求到期：该请求不能报成功，交给等待方判超时
            if now > entry.deadline:
                entry.future.set_exception(
                    TimeoutError("request deadline exceeded before result ready")
                )
            else:
                entry.future.set_result((float(score), batch_no))
