"""验证 FAIL_BATCH_NO 环境变量注入的一次性故障路径。

面向以 FAIL_BATCH_NO=N 启动的实例（脚本内默认 N=2）：
  * 前 N-1 批正常 200；
  * 第 N 批整批 500（只一次）；
  * 第 N+1 批起恢复 200，failed_batches 恒为 1。

用法： BASE_URL=http://127.0.0.1:8001 python tests/test_env_failure.py
"""
from __future__ import annotations

import asyncio
import os

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8001")


def body(disk_no: int) -> dict:
    return {
        "disk_no": disk_no,
        "temperature": 40.0,
        "reallocated_sectors": 1,
        "pending_sectors": 0,
        "deadline_ms": 5000,
    }


async def one_batch(client: httpx.AsyncClient, tag: int) -> list[int]:
    rs = await asyncio.gather(
        *[client.post(f"{BASE_URL}/predict", json=body(tag * 10 + i)) for i in range(4)]
    )
    return [r.status_code for r in rs]


async def main() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        cfg = (await client.get(f"{BASE_URL}/stats")).json()
        n = cfg["config"]["fail_batch_no"]
        assert n > 0, "目标实例必须设置 FAIL_BATCH_NO>0"
        assert cfg["total_batches"] == 0, (
            f"实例已处理过 {cfg['total_batches']} 批，一次性故障可能已被消费，"
            f"请重启一个全新的 FAIL_BATCH_NO={n} 实例后再跑本测试"
        )
        print(f"target FAIL_BATCH_NO={n}")

        observed: list[list[int]] = []
        for tag in range(n + 2):  # n-1 正常批 + 1 故障批 + 2 恢复批
            observed.append(await one_batch(client, tag))
            await asyncio.sleep(0.03)

        stats = (await client.get(f"{BASE_URL}/stats")).json()
        print("batch status sequences:", observed)
        print("failed_batches:", stats["failed_batches"])

        ok = (
            all(codes == [200] * 4 for codes in observed[: n - 1])
            and observed[n - 1] == [500] * 4
            and all(codes == [200] * 4 for codes in observed[n:])
            and stats["failed_batches"] == 1
        )
        print("RESULT:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
