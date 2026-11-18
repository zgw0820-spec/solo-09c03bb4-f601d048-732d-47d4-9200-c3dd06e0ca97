"""回执存储层单元自测（直接调用 ReceiptStore，不需要启动 HTTP 服务）。

用法：
    python tests/test_receipts_store.py

覆盖：
 1. 0.5 边界：分数恰好 0.5 判定为高风险（误报），0.4999 不算
 2. 同一盘多条回执分别计样
 3. 重复反馈幂等、更正只认最新结论（revision 递增）
 4. 未知回执抛 ReceiptNotFoundError
 5. 重新打开同一数据库文件后数据仍在（进程级持久化）
 6. 并发写入 + 并发报告：样本数恒定、fp+fn 不大于样本数
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.receipts import (  # noqa: E402
    ReceiptNotFoundError,
    ReceiptStore,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name} {detail}")


def new_store() -> tuple[ReceiptStore, str]:
    path = tempfile.mktemp(prefix="receipt_store_test_", suffix=".sqlite3")
    store = ReceiptStore(path)
    store.initialize()
    return store, path


def main() -> None:
    store, path = new_store()
    try:
        # 1. 0.5 边界
        at = store.register(disk_no=1, risk_score=0.5, batch_id=1)
        below = store.register(disk_no=2, risk_score=0.4999, batch_id=1)
        store.submit_feedback(at.receipt_id, False)     # 高风险 + 正常 -> 误报
        store.submit_feedback(below.receipt_id, False)  # 低风险 + 正常 -> 不算
        rep = store.report()
        check(
            "0.5 边界：分数=0.5 判高风险，0.4999 不判",
            rep.false_positives == 1
            and rep.false_negatives == 0
            and rep.feedback_samples == 2
            and rep.high_risk_threshold == 0.5,
            f"{rep}",
        )

        # 2. 同盘多预测分别计样
        a = store.register(disk_no=9, risk_score=0.9, batch_id=2)
        b = store.register(disk_no=9, risk_score=0.1, batch_id=3)
        store.submit_feedback(a.receipt_id, True)
        store.submit_feedback(b.receipt_id, True)
        rep = store.report()
        check(
            "同一盘两条回执分别计样（高风险真阳性 + 低风险漏报）",
            rep.feedback_samples == 4 and rep.false_negatives == 1,
            f"{rep}",
        )

        # 3. 幂等 / 更正
        f1, c1 = store.submit_feedback(a.receipt_id, True)
        f2, c2 = store.submit_feedback(a.receipt_id, False)
        f3, c3 = store.submit_feedback(a.receipt_id, False)
        rep = store.report()
        record = store.get_receipt(a.receipt_id)
        check(
            "重复反馈幂等、更正递增 revision 且报告只认最新结论",
            (f1.revision, c1) == (1, False)
            and (f2.revision, c2) == (2, True)
            and (f3.revision, c3) == (2, False)
            and record.feedback is not None
            and record.feedback.actual_abnormal is False
            and rep.false_positives == 2
            and rep.false_negatives == 1
            and rep.feedback_samples == 4,
            f"f1={f1} f2={f2} f3={f3} report={rep}",
        )

        # 4. 未知回执
        unknowns = 0
        for op in (
            lambda: store.get_receipt("nope"),
            lambda: store.submit_feedback("nope", True),
        ):
            try:
                op()
            except ReceiptNotFoundError:
                unknowns += 1
        check("未知回执查询/反馈抛 ReceiptNotFoundError", unknowns == 2)

        # 5. 并发写入 + 并发报告
        n = 12
        ids: list[str | None] = [None] * n
        stop = threading.Event()
        problems: list[dict] = []
        base = store.report()

        def writer(i: int) -> None:
            r = store.register(disk_no=100 + i, risk_score=0.8, batch_id=10 + i)
            ids[i] = r.receipt_id
            store.submit_feedback(r.receipt_id, i % 2 == 0)

        def observer() -> None:
            while not stop.is_set():
                rep = store.report()
                if rep.feedback_samples < base.feedback_samples:
                    problems.append({"samples": rep.feedback_samples})
                if rep.false_positives + rep.false_negatives > rep.feedback_samples:
                    problems.append({"over": rep.feedback_samples})

        obs = [threading.Thread(target=observer) for _ in range(2)]
        for t in obs:
            t.start()
        writers = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in obs:
            t.join()
        rep = store.report()
        check(
            "并发写入+报告：样本数恒定、无重复计入",
            not problems
            and rep.feedback_samples == base.feedback_samples + n
            and rep.false_positives == base.false_positives + n // 2
            and len(set(ids)) == n,
            f"problems={len(problems)} report={rep}",
        )

        store.close()

        # 6. 重新打开同一文件
        reopened = ReceiptStore(path)
        reopened.initialize()
        try:
            record = reopened.get_receipt(a.receipt_id)
            rep2 = reopened.report()
            check(
                "重新打开数据库文件：回执/反馈/报告保持一致",
                record.feedback is not None
                and record.feedback.revision == 2
                and rep2 == rep,
                f"{rep2}",
            )
        finally:
            reopened.close()
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except OSError:
                pass

    print("\n==== 汇总 ====")
    failed = [r for r in results if r[0] == FAIL]
    for status, name, detail in results:
        print(f"{status:4} {name} {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
