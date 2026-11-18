"""关键自测脚本（不依赖 pytest，直接 asyncio + httpx 运行）。

用法：
    BASE_URL=http://127.0.0.1:8000 python tests/selftest.py

覆盖：
 1. 单请求：分数区间、批次号、可复现
 2. 合批：12 个并发请求合成 <=8 的批次，窗口 <=40ms
 3. 独立期限：同批短期限请求 504，长期限请求仍 200，不被拖累
 4. 排队名额：超过 32 名额出现 503，且服务随后恢复
 5. 一次性模型故障：/debug/arm-failure 武装后该批 500，后续批恢复 200
 6. 断连腾名额：客户端提前断开后队列深度回落
 7. 排队超时腾名额：满员 503、短期限排队超时 504，之后队列清空

注：本脚本面向 FAIL_BATCH_NO=0 的实例运行（故障由 arm 端点确定性触发）；
FAIL_BATCH_NO 环境变量注入路径由 tests/test_env_failure.py 单独验证。
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}")


def payload(disk_no: int, deadline_ms: int = 5000, **over) -> dict:
    body = {
        "disk_no": disk_no,
        "temperature": 42.0,
        "reallocated_sectors": 10,
        "pending_sectors": 2,
        "deadline_ms": deadline_ms,
    }
    body.update(over)
    return body


async def stats(client: httpx.AsyncClient) -> dict:
    return (await client.get(f"{BASE_URL}/stats")).json()


async def test_single(client: httpx.AsyncClient) -> None:
    r1 = await client.post(f"{BASE_URL}/predict", json=payload(1))
    r2 = await client.post(f"{BASE_URL}/predict", json=payload(1))
    ok = (
        r1.status_code == 200
        and 0.0 <= r1.json()["risk_score"] <= 1.0
        and isinstance(r1.json()["batch_id"], int)
        and r1.json()["risk_score"] == r2.json()["risk_score"]
    )
    check("单请求 200/分数0~1/固定权重可复现", ok, f"resp={r1.text}")


async def test_batching(client: httpx.AsyncClient) -> None:
    before = await stats(client)
    n = 12
    t0 = time.monotonic()
    rs = await asyncio.gather(
        *[client.post(f"{BASE_URL}/predict", json=payload(100 + i)) for i in range(n)]
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    after = await stats(client)
    batch_ids = {r.json()["batch_id"] for r in rs if r.status_code == 200}
    ok_codes = all(r.status_code == 200 for r in rs)
    # 12 个请求、单批<=8 -> 至少 2 批；每批人数<=8（由服务端配置保证，这里核对批数增量）
    batches_created = after["total_batches"] - before["total_batches"]
    inf_created = after["total_inferences"] - before["total_inferences"]
    ok = (
        ok_codes
        and len(batch_ids) >= 2
        and batches_created >= 2
        and inf_created == n
        and elapsed_ms < 400  # 两批推理(2x120ms)+窗口余量
    )
    check(
        "12 并发合批(<=8/批)且一次矩阵推理",
        ok,
        f"batches={batches_created}, inferences={inf_created}, "
        f"batch_ids={sorted(batch_ids)}, elapsed_ms={elapsed_ms:.1f}",
    )


async def test_window(client: httpx.AsyncClient) -> None:
    # 两个错开 15ms 的请求，窗口 40ms 内应同批
    async def call(i: int, delay: float) -> tuple[int, int]:
        await asyncio.sleep(delay)
        r = await client.post(f"{BASE_URL}/predict", json=payload(200 + i))
        return r.status_code, r.json()["batch_id"]

    a, b = await asyncio.gather(call(0, 0.0), call(1, 0.015))
    check("组批窗口 40ms 内同批", a[0] == b[0] == 200 and a[1] == b[1], f"{a} {b}")


async def test_deadline_isolation(client: httpx.AsyncClient) -> None:
    # 模拟延迟 120ms：短期限 30ms 必超时，长期限 2s 必成功，二者同批
    async def call(deadline: int) -> httpx.Response:
        return await client.post(
            f"{BASE_URL}/predict", json=payload(300, deadline_ms=deadline)
        )

    short, long_ = await asyncio.gather(call(30), call(2000))
    check(
        "同批独立期限：短期限504不拖累长期限200",
        short.status_code == 504 and long_.status_code == 200,
        f"short={short.status_code} long={long_.status_code} {long_.text}",
    )
    # 过期结果绝不能报成功
    check("超时响应体不含风险分数", "risk_score" not in short.text, short.text)


async def test_queue_limit(client: httpx.AsyncClient) -> None:
    # 队列 32 + 批内 8，一次性压 60 个，必然出现 503
    rs = await asyncio.gather(
        *[
            client.post(
                f"{BASE_URL}/predict",
                json=payload(400 + i, deadline_ms=8000),
            )
            for i in range(60)
        ],
        return_exceptions=True,
    )
    codes = [r.status_code if isinstance(r, httpx.Response) else -1 for r in rs]
    n503 = codes.count(503)
    # 压测后服务应恢复
    recv = await client.post(f"{BASE_URL}/predict", json=payload(999))
    depth = (await stats(client))["queue_depth"]
    check(
        "排队上限32：超额503且服务恢复、名额释放",
        n503 > 0 and recv.status_code == 200 and depth == 0,
        f"503x{n503}, recover={recv.status_code}, depth={depth}",
    )


async def test_model_failure(client: httpx.AsyncClient) -> None:
    before = await stats(client)
    # 确定性武装：下一次矩阵推理失败一次
    arm = await client.post(f"{BASE_URL}/debug/arm-failure")
    assert arm.status_code == 200, arm.text

    # 武装后的第一批应整批 500
    failed_rs = await asyncio.gather(
        *[client.post(f"{BASE_URL}/predict", json=payload(500 + i)) for i in range(8)]
    )
    failed_codes = [r.status_code for r in failed_rs]
    await asyncio.sleep(0.05)

    # 后续批必须恢复 200，证明只影响该批
    ok_rs = await asyncio.gather(
        *[client.post(f"{BASE_URL}/predict", json=payload(550 + i)) for i in range(8)]
    )
    ok_codes = [r.status_code for r in ok_rs]
    after = await stats(client)
    check(
        "武装故障：该批500且后续批恢复200",
        failed_codes == [500] * 8
        and ok_codes == [200] * 8
        and after["failed_batches"] == before["failed_batches"] + 1,
        f"failed={failed_codes}, recovered={ok_codes}, "
        f"failed_batches={after['failed_batches']}",
    )


async def test_disconnect() -> None:
    # 裸 HTTP：发出请求后立即断开，核对服务端队列深度回落（名额腾出）
    from urllib.parse import urlsplit

    host = urlsplit(BASE_URL).hostname or "127.0.0.1"
    port = urlsplit(BASE_URL).port or 80

    async def open_and_close() -> None:
        reader, writer = await asyncio.open_connection(host, port)
        body = json.dumps(payload(600, deadline_ms=5000)).encode()
        writer.write(
            b"POST /predict HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        await asyncio.sleep(0.01)
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, BrokenPipeError):
            pass

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=8.0) as client:
        depth_before = (await stats(client))["queue_depth"]
        await open_and_close()
        # 等待轮询(20ms)感知断连并 discard
        await asyncio.sleep(0.4)
        depth_after = (await stats(client))["queue_depth"]
    check(
        "断连腾出排队名额",
        depth_after == depth_before == 0,
        f"before={depth_before} after={depth_after}",
    )


async def test_queue_timeout_releases_slot(client: httpx.AsyncClient) -> None:
    # 占满处理能力并填满排队(8 在批 + 32 排队 = 硬容量 40)，再发探针：
    # 用 60 个请求施压（硬容量 40），轮询到队列满 32 的瞬间发探针。
    # 满载为瞬态（每 ~120ms 抽走 8 个），探针若恰好撞上抽批则整轮重试（有界 3 次）。
    depth_full = -1
    r503: httpx.Response | None = None
    for attempt in range(3):
        blockers = [
            asyncio.create_task(
                client.post(
                    f"{BASE_URL}/predict", json=payload(800 + attempt * 100 + i, deadline_ms=8000)
                )
            )
            for i in range(60)
        ]
        wait_until = time.monotonic() + 2.0
        while time.monotonic() < wait_until:
            depth_full = (await stats(client))["queue_depth"]
            if depth_full == 32:
                break
            await asyncio.sleep(0.002)
        r503 = await client.post(
            f"{BASE_URL}/predict", json=payload(777, deadline_ms=5000)
        )
        await asyncio.gather(*blockers, return_exceptions=True)
        if r503.status_code == 503:
            break
    assert r503 is not None
    check(
        "队列满员超额请求被拒(503)",
        depth_full == 32 and r503.status_code == 503,
        f"depth_full={depth_full}, status={r503.status_code}",
    )

    # 等一批处理完腾出队列名额后，发一个期限极短、注定排队超时的请求
    await asyncio.sleep(0.15)
    r504 = await client.post(
        f"{BASE_URL}/predict", json=payload(778, deadline_ms=5)
    )
    check(
        "排队等不到处理则独立期限超时(504)且不报成功",
        r504.status_code == 504 and "risk_score" not in r504.text,
        f"status={r504.status_code}, body={r504.text}",
    )

    await asyncio.gather(*blockers, return_exceptions=True)
    await asyncio.sleep(0.2)
    depth_after = (await stats(client))["queue_depth"]
    check("超时/拒绝后名额全部释放，队列清空", depth_after == 0, f"depth={depth_after}")


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        health = await client.get("/health")
        assert health.status_code == 200, health.text
        print("config:", (await stats(client))["config"], "\n")

        await test_single(client)
        await test_batching(client)
        await test_window(client)
        await test_deadline_isolation(client)
        await test_queue_limit(client)
        await test_model_failure(client)
        await test_queue_timeout_releases_slot(client)
    await test_disconnect()

    print("\n==== 汇总 ====")
    failed = [r for r in results if r[0] == FAIL]
    for status, name, detail in results:
        print(f"{status:4} {name} {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
