"""服务配置：全部通过环境变量注入，便于 Compose 编排与自测。"""
from __future__ import annotations

import os


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


class Settings:
    # —— 组批参数 ——
    max_batch_size: int = _get_int("MAX_BATCH_SIZE", 8)
    # 组批窗口（毫秒）
    batch_window_ms: float = _get_float("BATCH_WINDOW_MS", 40.0)
    # 排队名额上限
    max_queue: int = _get_int("MAX_QUEUE", 32)

    # —— 模拟推理延迟（秒），每次矩阵推理固定 sleep 该时长 ——
    model_latency_s: float = _get_float("MODEL_LATENCY_S", 0.12)
    # 一次性故障：第 N 次矩阵推理（批）抛错；<=0 表示不注入
    fail_batch_no: int = _get_int("FAIL_BATCH_NO", 0)

    # —— 回执与巡检反馈（SQLite）——
    # 数据库文件路径；默认放在服务工作目录的 data/ 下，Compose 中挂载为持久卷
    receipt_db_path: str = os.getenv("RECEIPT_DB_PATH", "data/receipts.sqlite3")
    # 单次 SQLite 等待锁的超时（毫秒）
    receipt_db_busy_timeout_ms: int = _get_int("RECEIPT_DB_BUSY_TIMEOUT_MS", 1000)
    # 单次回执/反馈写入的等待上限（秒）；超时视为回执库暂不可用，
    # 预测仍正常给分（只是不回执），反馈请求返回 503 由运维重试
    receipt_write_timeout_s: float = _get_float("RECEIPT_WRITE_TIMEOUT_S", 1.0)

    @property
    def batch_window_s(self) -> float:
        return self.batch_window_ms / 1000.0


settings = Settings()
