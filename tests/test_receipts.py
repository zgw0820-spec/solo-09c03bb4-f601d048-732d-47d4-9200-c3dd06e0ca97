"""回执与巡检反馈自测（不依赖 pytest，直接 asyncio + httpx 运行）。

用法：
    BASE_URL=http://127.0.0.1:8000 python tests/test_receipts.py

要求目标实例的 FAIL_BATCH_NO=0（本脚本自行用 /debug/arm-failure 注入故障），
且使用一个全新的回执库（脚本会核对初始已反馈样本数为 0）。

覆盖：
 1. 只有期限内成功的 200 响应才带回执；504 / 500 均不带回执、不落库
 2. 回执可查询：分数、盘号、批次号与预测响应一致
 3. 同一盘不同预测分别计样；重复反馈幂等（样本数不重复计）
 4. 更正只认最新结论（误报 / 漏报随最新反馈迁移）
 5. 0.5 边界：0.5 及以上算高风险（>=0.5）
 6. 未知回执：查询 404、反馈 404 且不产生样本
 7. 故障批按原期限应答 500，但不发放回执
 8. 并发更正与报告查询：样本数恒定、计数不混用新旧记录
 9. 并发预测：每笔 200 都有唯一回执，回执数与成功数一致

重启后回执/反馈仍有效由 tests/test_restart_persistence.py 单独验证。
"""
from __future__ import annotations

import asyncio
import os

import httpx

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}")


async def stats(client: httpx.AsyncClient) -> dict:
    return (await client.get(f"{BASE_URL}/stats")).json()


async def report(client: httpx.AsyncClient) -> dict:
    r = await client.get(f"{BASE_URL}/feedback/report")
    assert r.status_code == 200, r.text
    return r.json()


def predict_body(
    disk_no: int,
    *,
    temperature: float = 42.0,
    reallocated_sectors: int = 10,
    pending_sectors: int = 2,
    deadline_ms: int = 5000,
) -> dict:
    return {
        "disk_no": disk_no,
        "temperature": temperature,
        "reallocated_sectors": reallocated_sectors,
        "pending_sectors": pending_sectors,
        "deadline_ms": deadline_ms,
    }


# 固定模型下的已知分数（<0.5 低风险 / >0.5 高风险）
LOW = dict(temperature=42.0, reallocated_sectors=10, pending_sectors=2)     # ~0.20
HIGH = dict(temperature=50.0, reallocated_sectors=100, pending_sectors=30)  # ~0.95


async def predict(client: httpx.AsyncClient, disk_no: int, **kw) -> httpx.Response:
    return await client.post(f"{BASE_URL}/predict", json=predict_body(disk_no, **kw))


async def feedback(
    client: httpx.AsyncClient, receipt_id: str, actual_abnormal: bool
) -> httpx.Response:
    return await client.post(
        f"{BASE_URL}/receipts/{receipt_id}/feedback",
        json={"actual_abnormal": actual_abnormal},
    )


async def receipt(client: httpx.AsyncClient, receipt_id: str) -> httpx.Response:
    return await client.get(f"{BASE_URL}/receipts/{receipt_id}")


async def test_zero_start(client: httpx.AsyncClient) -> None:
    s = await stats(client)
    store = s["receipt_store"]
    check(
        "回执库已就绪且为全新库（已反馈样本=0）",
        store["ready"] and store["feedback"] == 0,
        f"store={store}",
    )


async def test_receipt_only_on_success(client: httpx.AsyncClient) -> None:
    cfg_before = await stats(client)

    ok = await predict(client, 10, **LOW)
    body = ok.json()
    check(
        "200 成功预测发放回执且分数/盘号一致",
        ok.status_code == 200
        and isinstance(body.get("receipt_id"), str)
        and len(body["receipt_id"]) > 0
        and body["disk_no"] == 10,
        f"{ok.status_code} {body}",
    )

    # 期限超时：504 且无回执
    r504 = await predict(client, 11, deadline_ms=5, **LOW)
    check(
        "504 超时响应不带回执且不落库",
        r504.status_code == 504 and "receipt_id" not in r504.text,
        f"{r504.status_code} {r504.text}",
    )

    # 模型故障批：500 且无回执
    assert (await client.post(f"{BASE_URL}/debug/arm-failure")).status_code == 200
    failed = await predict(client, 12, **LOW)
    await asyncio.sleep(0.05)
    check(
        "故障批 500 不带回执且不落库",
        failed.status_code == 500 and "receipt_id" not in failed.text,
        f"{failed.status_code} {failed.text}",
    )

    cfg_after = await stats(client)
    delta = cfg_after["receipt_store"]["receipts"] - cfg_before["receipt_store"]["receipts"]
    check(
        "仅成功的 1 次预测产生 1 条回执记录",
        delta == 1,
        f"receipts delta={delta}",
    )

    view = await receipt(client, body["receipt_id"])
    v = view.json() if view.status_code == 200 else {}
    check(
        "按回执号可查询预测记录（分数/批次/盘号一致，尚无反馈）",
        view.status_code == 200
        and v["disk_no"] == body["disk_no"]
        and v["risk_score"] == body["risk_score"]
        and v["batch_id"] == body["batch_id"]
        and v["feedback"] is None,
        f"{view.status_code} {v}",
    )


async def test_sample_counting(client: httpx.AsyncClient) -> None:
    before = await report(client)
    # 同一盘两次预测（两条回执）+ 另一盘一次预测 -> 三个样本
    r1 = await predict(client, 20, **HIGH)
    r2 = await predict(client, 20, **HIGH)
    r3 = await predict(client, 21, **LOW)
    ids = [r1.json()["receipt_id"], r2.json()["receipt_id"], r3.json()["receipt_id"]]
    check("同一盘不同预测各自发回执", len(set(ids)) == 3, f"{ids}")

    for rid in ids:
        resp = await feedback(client, rid, True)
        assert resp.status_code == 200, resp.text
    after = await report(client)
    check(
        "同一盘不同预测分别计样，重复反馈不重复计数",
        after["feedback_samples"] == before["feedback_samples"] + 3
        and after["false_negatives"] == before["false_negatives"] + 1,
        f"before={before} after={after}",
    )

    # 对 r1 重复提交相同结论：不应改变任何数字或 revision
    dup = await feedback(client, ids[0], True)
    d = dup.json()
    after_dup = await report(client)
    check(
        "同一回执重复相同反馈：幂等（不重复计数、revision 不变）",
        dup.status_code == 200
        and d["corrected"] is False
        and d["revision"] == 1
        and after_dup == after,
        f"{dup.status_code} {d}",
    )

    # 更正：r1 是高风险且实际异常（不计误报/漏报），改为实际正常 -> 误报 +1；
    # r3 是低风险且实际异常（漏报），改为实际正常 -> 漏报 -1。样本数始终不变，
    # 报告只认每个回执的最新结论。
    fix1 = await feedback(client, ids[0], False)
    fix3 = await feedback(client, ids[2], False)
    latest = await report(client)
    f1, f3 = fix1.json(), fix3.json()
    check(
        "更正后报告只认最新结论（误报+1、漏报-1、样本数不变）",
        after["false_positives"] == before["false_positives"]
        and after["false_negatives"] == before["false_negatives"] + 1
        and fix1.status_code == fix3.status_code == 200
        and f1["corrected"] is True
        and f1["revision"] == 2
        and f3["corrected"] is True
        and latest["feedback_samples"] == after["feedback_samples"]
        and latest["false_negatives"] == after["false_negatives"] - 1
        and latest["false_positives"] == after["false_positives"] + 1,
        f"corrected={f1} latest={latest}",
    )

    view = (await receipt(client, ids[0])).json()
    check(
        "回执查询展示最新反馈（revision=2, 实际正常）",
        view["feedback"]["actual_abnormal"] is False
        and view["feedback"]["revision"] == 2,
        f"{view['feedback']}",
    )
    return latest


async def test_threshold_boundary(client: httpx.AsyncClient) -> None:
    """0.5 为高风险界线：>=0.5 预测高风险；0.5 及以下漏报不计。"""
    # 用已知固定分数：0.3726 (<0.5) 与 0.9501 (>=0.5)
    low = await predict(
        client, 30, temperature=46.5, reallocated_sectors=24, pending_sectors=6
    )
    high = await predict(
        client, 31, temperature=50.0, reallocated_sectors=100, pending_sectors=30
    )
    lb, hb = low.json(), high.json()
    check(
        "测试样本分数落在界线两侧",
        lb["risk_score"] < 0.5 <= hb["risk_score"],
        f"low={lb['risk_score']:.4f} high={hb['risk_score']:.4f}",
    )
    before = await report(client)
    await feedback(client, lb["receipt_id"], False)  # 低风险且正常：非误报
    await feedback(client, hb["receipt_id"], True)   # 高风险且异常：非漏报
    after = await report(client)
    check(
        "界线两侧正确分类：低风险正常不算误报、高风险异常不算漏报",
        after["feedback_samples"] == before["feedback_samples"] + 2
        and after["false_positives"] == before["false_positives"]
        and after["false_negatives"] == before["false_negatives"],
        f"before={before} after={after}",
    )
    check(
        "报告返回高风险界线 0.5",
        after["high_risk_threshold"] == 0.5,
        f"threshold={after['high_risk_threshold']}",
    )
    # 这两个样本也用于验证同一盘不同批次计样（同盘号不同预测）
    same_disk_a = await predict(client, 32, **LOW)
    same_disk_b = await predict(client, 32, **HIGH)
    a, b = same_disk_a.json(), same_disk_b.json()
    before2 = await report(client)
    await feedback(client, a["receipt_id"], False)
    await feedback(client, b["receipt_id"], False)
    after2 = await report(client)
    check(
        "同一盘一次低风险一次高风险：分别计样且仅高风险那次算误报",
        after2["feedback_samples"] == before2["feedback_samples"] + 2
        and after2["false_positives"] == before2["false_positives"] + 1,
        f"before={before2} after={after2}",
    )


async def test_unknown_receipt(client: httpx.AsyncClient) -> None:
    before = await report(client)
    missing = "0" * 32
    r_get = await receipt(client, missing)
    r_post = await feedback(client, missing, True)
    after = await report(client)
    check(
        "未知回执：查询/反馈均 404 且不产生样本",
        r_get.status_code == 404
        and r_post.status_code == 404
        and after == before,
        f"get={r_get.status_code} post={r_post.status_code}",
    )


async def test_failure_batch_no_receipt(client: httpx.AsyncClient) -> None:
    """故障批（模型 500）仍按原期限应答失败，不会返回分数或回执。"""
    assert (await client.post(f"{BASE_URL}/debug/arm-failure")).status_code == 200
    before = await stats(client)
    rs = await asyncio.gather(*[predict(client, 600 + i, **LOW) for i in range(8)])
    await asyncio.sleep(0.05)
    after = await stats(client)
    delta = after["receipt_store"]["receipts"] - before["receipt_store"]["receipts"]
    check(
        "故障批 8 个请求整批 500，均无回执、不落库",
        [r.status_code for r in rs] == [500] * 8
        and all("receipt_id" not in r.text for r in rs)
        and delta == 0,
        f"codes={[r.status_code for r in rs]} receipts_delta={delta}",
    )


async def test_concurrent_corrections(client: httpx.AsyncClient) -> None:
    """并发更正 + 并发报告：样本数恒定、计数不自相矛盾。

    构造 N 个高风险回执，反复在“实际异常(不计误报)”与“实际正常(误报)”
    之间更正，同时高频查询报告。SQLite 读事务 + 单条 JOIN 聚合保证每次报告
    是一个一致快照；本测试核对：已反馈样本数从不漂移（重复更正不重复计数）、
    误报+漏报不会超过样本数（新旧记录混用会超），停止更正后报告的数字与
    逐条回执查询独立推导的结果完全一致。
    """
    n = 6
    rs = await asyncio.gather(*[predict(client, 700 + i, **HIGH) for i in range(n)])
    ids = [r.json()["receipt_id"] for r in rs]
    for rid in ids:
        resp = await feedback(client, rid, True)
        assert resp.status_code == 200, resp.text

    before = await report(client)
    # 上面已为 6 个回执写入反馈，before 已包含它们
    expected_samples = before["feedback_samples"]
    base_fp = before["false_positives"]
    base_fn = before["false_negatives"]
    stop = asyncio.Event()
    problems: list[dict] = []

    async def churn(rid: str) -> None:
        flag = True
        while not stop.is_set():
            resp = await feedback(client, rid, flag)
            if resp.status_code != 200:
                problems.append({"feedback_status": resp.status_code})
                return
            flag = not flag

    async def observe() -> None:
        while not stop.is_set():
            rep = await report(client)
            if rep["feedback_samples"] != expected_samples:
                problems.append(rep)  # 重复更正被重复计数 / 样本数漂移
            elif rep["false_positives"] + rep["false_negatives"] > expected_samples:
                problems.append(rep)  # 新旧记录混用导致重复计入

    tasks = [asyncio.create_task(churn(rid)) for rid in ids]
    tasks += [asyncio.create_task(observe()) for _ in range(3)]
    await asyncio.sleep(1.2)
    stop.set()
    await asyncio.gather(*tasks)

    # 静止后：这 6 条回执按“最新结论”聚合，必须与 /feedback/report 完全一致。
    views = await asyncio.gather(*[receipt(client, rid) for rid in ids])
    latest = {
        v["receipt_id"]: (
            v["risk_score"] >= 0.5,
            v["feedback"]["actual_abnormal"],
        )
        for v in (r.json() for r in views)
    }
    fp_from_views = sum(1 for high, abnormal in latest.values() if high and not abnormal)
    fn_from_views = sum(1 for high, abnormal in latest.values() if not high and abnormal)
    final = await report(client)

    # 全部改回“实际异常”：6 条高风险样本都是真阳性，误报/漏报回到基线
    await asyncio.gather(*[feedback(client, rid, True) for rid in ids])
    restored = await report(client)

    check(
        "并发更正与报告查询：样本数恒定、无误报/漏报混用",
        not problems
        and final["feedback_samples"] == expected_samples
        and final["false_positives"] >= base_fp
        and final["false_positives"] - base_fp == fp_from_views
        and final["false_negatives"] - base_fn == fn_from_views
        and restored["feedback_samples"] == expected_samples
        and restored["false_positives"] == base_fp
        and restored["false_negatives"] == base_fn
        and all(v["feedback"]["revision"] >= 1 for v in (r.json() for r in views)),
        f"problems={len(problems)} final={final} restored={restored} "
        f"views_fp={fp_from_views} views_fn={fn_from_views}",
    )


async def test_concurrent_predictions(client: httpx.AsyncClient) -> None:
    before = await stats(client)
    n = 24
    rs = await asyncio.gather(*[predict(client, 900 + i, **LOW) for i in range(n)])
    after = await stats(client)
    ids = [r.json().get("receipt_id") for r in rs if r.status_code == 200]
    delta = after["receipt_store"]["receipts"] - before["receipt_store"]["receipts"]
    check(
        "并发预测：每个 200 一个唯一回执，回执数=成功数",
        all(r.status_code == 200 for r in rs)
        and len(ids) == n
        and len(set(ids)) == n
        and delta == n,
        f"ok={len(ids)} unique={len(set(ids))} receipts_delta={delta}",
    )


async def test_persistence_hint(client: httpx.AsyncClient) -> None:
    """留一条已反馈回执并打印，便于人工重启后核对（自动验证见
    tests/test_restart_persistence.py）。"""
    r = await predict(client, 1234, **LOW)
    rid = r.json()["receipt_id"]
    await feedback(client, rid, True)
    rep = await report(client)
    print(f"\n[persistence] receipt={rid} report={rep}")
    print("[persistence] 重启服务后执行：")
    print(f"  curl {BASE_URL}/receipts/{rid}")
    print(f"  curl {BASE_URL}/feedback/report")
    env_path = os.getenv("PERSIST_RECEIPT_FILE", "/tmp/disk_risk_persist_receipt.txt")
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"{rid}\n{rep['feedback_samples']}\n")
    except OSError:
        pass


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        assert (await client.get("/health")).status_code == 200
        await test_zero_start(client)
        await test_receipt_only_on_success(client)
        await test_sample_counting(client)
        await test_threshold_boundary(client)
        await test_unknown_receipt(client)
        await test_failure_batch_no_receipt(client)
        await test_concurrent_corrections(client)
        await test_concurrent_predictions(client)
        await test_persistence_hint(client)

    print("\n==== 汇总 ====")
    failed = [r for r in results if r[0] == FAIL]
    for status, name, detail in results:
        print(f"{status:4} {name} {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
