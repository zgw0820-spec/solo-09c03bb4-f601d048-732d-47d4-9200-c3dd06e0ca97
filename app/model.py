"""固定权重的 NumPy 逻辑回归磁盘风险模型。

特征（4 维，与请求字段一一对应）：
    温度、重映射扇区数、待映射扇区数、磁盘号
权重在模块加载时固定，不做任何训练 / 外部加载，保证随服务可复现。
输入先按固定均值/标准差做标准化，再执行一次矩阵运算 + sigmoid。
"""
from __future__ import annotations

import numpy as np

# —— 固定模型参数 ——
# 标准化统计量（来自离线巡检样本的固定值）
_MEAN = np.array([38.0, 12.0, 4.0, 256.0], dtype=np.float64)
_STD = np.array([10.0, 40.0, 16.0, 256.0], dtype=np.float64)

# 逻辑回归权重（列向量）与截距
_WEIGHTS = np.array([0.55, 0.90, 1.05, 0.08], dtype=np.float64).reshape(-1, 1)
_BIAS = -1.35


class ModelFailureError(RuntimeError):
    """模型处理失败（含注入的一次性故障）。"""


class LogisticRiskModel:
    def __init__(self, latency_s: float = 0.0, fail_batch_no: int = 0) -> None:
        self.latency_s = max(0.0, latency_s)
        # >0 时，第 fail_batch_no 次推理注入一次故障，之后自动复位不再触发
        self.fail_batch_no = fail_batch_no
        self._call_count = 0
        # 武装后，下一次推理必定失败一次（供测试/演示“一次故障”）
        self._armed_failure = False

    def arm_failure_once(self) -> None:
        """让下一次矩阵推理失败一次（测试/演示用途）。"""
        self._armed_failure = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """对一批特征执行一次矩阵推理，返回 0~1 风险分数（一维数组）。"""
        self._call_count += 1
        if self._armed_failure:
            self._armed_failure = False
            raise ModelFailureError(
                f"armed one-shot model failure on inference #{self._call_count}"
            )
        if self.fail_batch_no > 0 and self._call_count == self.fail_batch_no:
            raise ModelFailureError(
                f"injected model failure on inference #{self._call_count}"
            )

        x = features.astype(np.float64, copy=True)
        # 固定标准化；防止除零
        x = (x - _MEAN) / np.maximum(_STD, 1e-9)
        # 一次矩阵乘法：(n,4) @ (4,1) -> (n,1)
        logits = x @ _WEIGHTS + _BIAS
        scores = 1.0 / (1.0 + np.exp(-logits))
        return scores.reshape(-1)
