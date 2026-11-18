"""重启持久化自测：回执与巡检反馈在服务重启后仍有效。

自管一个 uvicorn 子进程（临时目录内的 SQLite 库），流程：
 1. 启动实例，成功预测取得回执，提交巡检反馈；
 2. 结束进程，用同一数据库文件重新启动；
 3. 核对回执可查、反馈仍在、/feedback/report 计数不变；
 4. 重启后继续更正反馈，报告按最新结论更新（仍只计一个样本）。

用法：
    python tests/test_restart_persistence.py
可选环境变量：
    PERSIST_TEST_PORT（默认 8010）、PERSIST_TEST_DB_DIR（默认临时目录）
"""
from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import httpx

PORT = int(os.getenv("PERSIST_TEST_PORT", "8010"))
BASE_URL = f"http://127.0.0.1:{PORT}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}")


def wait_healthy(timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=0.5):
                pass
            resp = httpx.get(f"{BASE_URL}/health", timeout=2.0)
            if resp.status_code == 200:
                return
        except (OSError, httpx.HTTPError):
            pass
        time.sleep(0.15)
    raise RuntimeError("server did not become healthy")


def start_server(db_path: str) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "RECEIPT_DB_PATH": db_path,
            "MODEL_LATENCY_S": "0.01",
            "FAIL_BATCH_NO": "0",
            "PYTHONPATH": ROOT,
        }
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_healthy()
    except Exception:
        proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"startup failed:\n{out}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def body(disk_no: int, high: bool) -> dict:
    return {
        "disk_no": disk_no,
        "temperature": 50.0 if high else 42.0,
        "reallocated_sectors": 100 if high else 10,
        "pending_sectors": 30 if high else 2,
        "deadline_ms": 5000,
    }


async def main() -> None:
    db_dir = os.getenv("PERSIST_TEST_DB_DIR") or tempfile.mkdtemp(prefix="disk_risk_persist_")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "receipts.sqlite3")

    proc = start_server(db_path)
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            high = (await client.post("/predict", json=body(1, high=True))).json()
            low = (await client.post("/predict", json=body(1, high=False))).json()
            rid_high, rid_low = high["receipt_id"], low["receipt_id"]
            # 高风险 + 实际异常（真阳性）；低风险 + 实际异常（漏报）
            for rid, abnormal in ((rid_high, True), (rid_low, True)):
                r = await client.post(
                    f"/receipts/{rid}/feedback", json={"actual_abnormal": abnormal}
                )
                assert r.status_code == 200, r.text
            report_before = (await client.get("/feedback/report")).json()
            check(
                "重启前：回执与反馈已落库",
                rid_high and rid_low and report_before["feedback_samples"] == 2,
                f"report={report_before}",
            )
    finally:
        stop_server(proc)

    # —— 重启：同一数据库文件 ——
    proc = start_server(db_path)
    try:
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
            report_after = (await client.get("/feedback/report")).json()
            view = (await client.get(f"/receipts/{rid_high}")).json()
            health = (await client.get("/health")).json()
            check(
                "重启后：回执仍可查、反馈结论保留",
                view["receipt_id"] == rid_high
                and view["feedback"] is not None
                and view["feedback"]["actual_abnormal"] is True
                and view["feedback"]["revision"] == 1,
                f"{view}",
            )
            check(
                "重启后：误报/漏报/样本数与重启前一致",
                report_after == report_before,
                f"before={report_before} after={report_after}",
            )
            check(
                "重启后：回执库就绪、合批指标归零（进程内状态）",
                health["receipt_store_ready"] is True,
                f"{health}",
            )

            # 重启后继续更正：仍只认最新结论且样本数不变
            fix = await client.post(
                f"/receipts/{rid_high}/feedback", json={"actual_abnormal": False}
            )
            corrected = fix.json()
            report_fixed = (await client.get("/feedback/report")).json()
            view_fixed = (await client.get(f"/receipts/{rid_high}")).json()
            check(
                "重启后更正：revision 递增、报告只认最新结论",
                corrected["corrected"] is True
                and corrected["revision"] == 2
                and view_fixed["feedback"]["revision"] == 2
                and report_fixed["feedback_samples"] == report_before["feedback_samples"]
                and report_fixed["false_positives"]
                == report_before["false_positives"] + 1,
                f"corrected={corrected} report={report_fixed}",
            )
    finally:
        stop_server(proc)

    if not os.getenv("PERSIST_TEST_DB_DIR"):
        shutil.rmtree(db_dir, ignore_errors=True)

    print("\n==== 汇总 ====")
    failed = [r for r in results if r[0] == FAIL]
    for status, name, detail in results:
        print(f"{status:4} {name} {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
