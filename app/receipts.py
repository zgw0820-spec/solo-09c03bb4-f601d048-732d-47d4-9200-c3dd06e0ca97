"""SQLite 回执与巡检反馈存储。

设计要点：
* 只有拿到成功分数（请求期限内、HTTP 200）的预测才登记回执；
* 回执号在一次预测内唯一，运维凭回执号提交“该盘实际是否异常”；
* 每个回执最多一条反馈记录，重复提交幂等，更正覆盖旧结论（revision 递增）；
* 误报 / 漏报统计只读取“每个回执的最新结论”，单条 JOIN 聚合语句在
  一个读事务内完成，查询与并发的更正不会混用新旧记录；
* 写操作走专用写线程 + 组提交（group commit）：同一组写入共享一次
  BEGIN IMMEDIATE / COMMIT，避免 FULL 同步下每条回执一次 fsync 拖慢合批响应；
  每个写操作包在 SAVEPOINT 中，个别操作的业务错误（如回执不存在）只回滚
  自己，不影响同组其它写入；
* SQLite 使用 WAL + synchronous=FULL，进程重启后回执与反馈仍在。
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import os
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, TypeVar

logger = logging.getLogger("disk_risk.receipts")

# 高风险界线：分数 >= 0.5 视为高风险（需求固定值，不做运行期配置）
HIGH_RISK_THRESHOLD = 0.5

_GROUP_COMMIT_MAX = 256
_SHUTDOWN = object()

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS prediction_receipts (
        receipt_id      TEXT PRIMARY KEY,
        disk_no         INTEGER NOT NULL,
        risk_score      REAL    NOT NULL,
        batch_id        INTEGER NOT NULL,
        created_at_ms   INTEGER NOT NULL,
        created_at      TEXT    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS inspection_feedback (
        receipt_id      TEXT PRIMARY KEY,
        actual_abnormal INTEGER NOT NULL CHECK (actual_abnormal IN (0, 1)),
        revision        INTEGER NOT NULL DEFAULT 1,
        created_at_ms   INTEGER NOT NULL,
        updated_at_ms   INTEGER NOT NULL,
        created_at      TEXT    NOT NULL,
        updated_at      TEXT    NOT NULL,
        FOREIGN KEY (receipt_id) REFERENCES prediction_receipts(receipt_id)
    )
    """,
)


class ReceiptStoreUnavailable(RuntimeError):
    """回执库不可用（初始化失败、读写异常等）。"""


class ReceiptNotFoundError(LookupError):
    """回执号不存在。"""


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    disk_no: int
    risk_score: float
    batch_id: int
    created_at: str


@dataclass(frozen=True)
class Feedback:
    receipt_id: str
    actual_abnormal: bool
    revision: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ReceiptRecord:
    receipt: Receipt
    feedback: Feedback | None


@dataclass(frozen=True)
class Report:
    high_risk_threshold: float
    feedback_samples: int
    false_positives: int
    false_negatives: int


T = TypeVar("T")


def _utc_iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


class _WriteOp:
    """提交给写线程的一次写操作。"""

    __slots__ = ("fn", "done")

    def __init__(self, fn: Callable[[sqlite3.Connection], T]) -> None:
        self.fn = fn
        self.done: concurrent.futures.Future[T] = concurrent.futures.Future()


class ReceiptStore:
    """SQLite 回执 / 反馈存储。

    写操作统一由后台写线程串行执行并组提交；读操作使用独立连接，
    依赖 WAL 的读写并发，每次读都在自己的读事务内完成。
    所有公开方法都是阻塞调用，路由层通过 asyncio.to_thread 调用。
    """

    def __init__(
        self,
        db_path: str,
        busy_timeout_ms: int = 1000,
        write_timeout_s: float = 30.0,
    ) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = max(1, int(busy_timeout_ms))
        self.write_timeout_s = max(0.1, float(write_timeout_s))
        self._schema_lock = threading.Lock()
        self._ready = False
        self._last_error: str | None = None
        self._last_init_attempt = 0.0

        self._write_q: queue.Queue = queue.Queue()
        self._writer: threading.Thread | None = None
        self._writer_lock = threading.Lock()

    # ---------- 状态 ----------
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # ---------- 初始化 / 连接 ----------
    def initialize(self) -> None:
        """创建数据库目录与表结构并启动写线程；失败可重复调用重试。

        初始化失败后 1 秒内的再次调用直接快速失败（冷却），避免回执库长期
        不可用时每个请求都重复付出建库/等锁的代价；冷却结束后自动重试，
        因此目录/权限恢复后无需重启即可继续发回执。
        """
        with self._schema_lock:
            if self._ready:
                return
            now = time.monotonic()
            if self._last_error is not None and now - self._last_init_attempt < 1.0:
                raise ReceiptStoreUnavailable(self._last_error)
            self._last_init_attempt = now
            try:
                self._create_schema()
            except (sqlite3.Error, OSError) as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "receipt store initialization failed: %s", self._last_error
                )
                raise ReceiptStoreUnavailable(self._last_error) from exc
            self._ensure_writer()
            self._ready = True
            self._last_error = None
            logger.info("receipt store ready: %s", os.path.abspath(self.db_path))

    def close(self) -> None:
        """停止写线程并关闭其连接（服务退出时调用）。"""
        with self._writer_lock:
            writer, self._writer = self._writer, None
        if writer is not None and writer.is_alive():
            self._write_q.put(_SHUTDOWN)
            writer.join(timeout=5.0)
        self._ready = False

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.initialize()

    def _connect(self) -> sqlite3.Connection:
        directory = os.path.dirname(os.path.abspath(self.db_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,  # 自行管理事务
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    def _create_schema(self) -> None:
        conn = self._connect()
        try:
            # WAL 持久化在库文件中，只需初始化时设置
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in _SCHEMA:
                    conn.execute(statement)
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @staticmethod
    @contextlib.contextmanager
    def _read_tx(conn: sqlite3.Connection):
        """只读事务：单条聚合语句读到的是一个一致快照。"""
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        else:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("COMMIT")

    # ---------- 写线程（组提交） ----------
    def _ensure_writer(self) -> None:
        with self._writer_lock:
            if self._writer is not None and self._writer.is_alive():
                return
            thread = threading.Thread(
                target=self._writer_loop, name="receipt-writer", daemon=True
            )
            thread.start()
            self._writer = thread

    def _submit(self, fn: Callable[[sqlite3.Connection], T]) -> T:
        if not self._ready:
            self.initialize()
        self._ensure_writer()
        op = _WriteOp(fn)
        self._write_q.put(op)
        try:
            return op.done.result(timeout=self.write_timeout_s)
        except concurrent.futures.TimeoutError as exc:
            raise ReceiptStoreUnavailable("receipt store write timed out") from exc

    def _writer_loop(self) -> None:
        conn: sqlite3.Connection | None = None
        while True:
            item = self._write_q.get()
            if item is _SHUTDOWN:
                break
            group = [item]
            stop = False
            while len(group) < _GROUP_COMMIT_MAX:
                try:
                    nxt = self._write_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is _SHUTDOWN:
                    stop = True
                    break
                group.append(nxt)

            try:
                if conn is None:
                    conn = self._connect()
                results = self._run_group(conn, group)
            except BaseException as exc:  # noqa: BLE001 - 写线程不能静默死掉
                if conn is not None:
                    with contextlib.suppress(sqlite3.Error):
                        conn.execute("ROLLBACK")
                    with contextlib.suppress(sqlite3.Error):
                        conn.close()
                    conn = None
                err = ReceiptStoreUnavailable(
                    f"receipt store write failed: {type(exc).__name__}: {exc}"
                )
                for op in group:
                    if not op.done.done():
                        op.done.set_exception(err)
            else:
                for op, value, exc in results:
                    if exc is not None:
                        op.done.set_exception(exc)
                    else:
                        op.done.set_result(value)

            if stop:
                break

        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    def _run_group(
        self, conn: sqlite3.Connection, group: list[_WriteOp]
    ) -> list[tuple[_WriteOp, object, BaseException | None]]:
        results: list[tuple[_WriteOp, object, BaseException | None]] = []
        conn.execute("BEGIN IMMEDIATE")
        committed = False
        try:
            for idx, op in enumerate(group):
                savepoint = f"op_{idx}"
                conn.execute(f"SAVEPOINT {savepoint}")
                try:
                    value = op.fn(conn)
                except sqlite3.Error as exc:
                    conn.execute(f"ROLLBACK TO {savepoint}")
                    conn.execute(f"RELEASE {savepoint}")
                    results.append(
                        (
                            op,
                            None,
                            ReceiptStoreUnavailable(
                                f"receipt store write failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                    )
                except BaseException as exc:  # noqa: BLE001 - 业务错误逐个回滚
                    conn.execute(f"ROLLBACK TO {savepoint}")
                    conn.execute(f"RELEASE {savepoint}")
                    results.append((op, None, exc))
                else:
                    conn.execute(f"RELEASE {savepoint}")
                    results.append((op, value, None))
            conn.execute("COMMIT")
            committed = True
        finally:
            if not committed:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
        return results

    # ---------- 回执 ----------
    def counts(self) -> dict[str, int]:
        """回执 / 反馈记录总数（运维与自测核对用）。"""
        try:
            self._ensure_ready()
            conn = self._connect()
            try:
                with self._read_tx(conn):
                    row = conn.execute(
                        "SELECT "
                        " (SELECT COUNT(*) FROM prediction_receipts) AS receipts, "
                        " (SELECT COUNT(*) FROM inspection_feedback) AS feedback"
                    ).fetchone()
            finally:
                conn.close()
        except ReceiptStoreUnavailable:
            raise
        except (sqlite3.Error, OSError) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise ReceiptStoreUnavailable(
                f"receipt count query failed: {self._last_error}"
            ) from exc
        return {"receipts": int(row["receipts"]), "feedback": int(row["feedback"])}

    def register(self, *, disk_no: int, risk_score: float, batch_id: int) -> Receipt:
        """为一次成功预测登记回执，返回唯一回执号。"""
        receipt_id = uuid.uuid4().hex
        now_ms = int(time.time() * 1000)
        created_at = _utc_iso(now_ms)

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO prediction_receipts "
                "(receipt_id, disk_no, risk_score, batch_id, created_at_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    receipt_id,
                    disk_no,
                    float(risk_score),
                    batch_id,
                    now_ms,
                    created_at,
                ),
            )

        self._submit(write)
        return Receipt(
            receipt_id=receipt_id,
            disk_no=disk_no,
            risk_score=float(risk_score),
            batch_id=batch_id,
            created_at=created_at,
        )

    def get_receipt(self, receipt_id: str) -> ReceiptRecord:
        """按回执号读取回执与当前（最新）反馈。"""
        try:
            self._ensure_ready()
            conn = self._connect()
            try:
                with self._read_tx(conn):
                    row = conn.execute(
                        "SELECT p.receipt_id, p.disk_no, p.risk_score, p.batch_id, "
                        "       p.created_at, "
                        "       f.actual_abnormal, f.revision, "
                        "       f.created_at AS feedback_created_at, "
                        "       f.updated_at AS feedback_updated_at "
                        "FROM prediction_receipts p "
                        "LEFT JOIN inspection_feedback f "
                        "  ON f.receipt_id = p.receipt_id "
                        "WHERE p.receipt_id = ?",
                        (receipt_id,),
                    ).fetchone()
            finally:
                conn.close()
        except ReceiptStoreUnavailable:
            raise
        except (sqlite3.Error, OSError) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise ReceiptStoreUnavailable(
                f"read receipt failed: {self._last_error}"
            ) from exc

        if row is None:
            raise ReceiptNotFoundError(f"receipt {receipt_id!r} not found")

        receipt = Receipt(
            receipt_id=row["receipt_id"],
            disk_no=row["disk_no"],
            risk_score=row["risk_score"],
            batch_id=row["batch_id"],
            created_at=row["created_at"],
        )
        feedback = None
        if row["actual_abnormal"] is not None:
            feedback = Feedback(
                receipt_id=row["receipt_id"],
                actual_abnormal=bool(row["actual_abnormal"]),
                revision=row["revision"],
                created_at=row["feedback_created_at"],
                updated_at=row["feedback_updated_at"],
            )
        return ReceiptRecord(receipt=receipt, feedback=feedback)

    # ---------- 巡检反馈 ----------
    def submit_feedback(
        self, receipt_id: str, actual_abnormal: bool
    ) -> tuple[Feedback, bool]:
        """提交/更正某回执的实际巡检结论。

        返回 (最新反馈, 是否发生了结论更正)。同一结论重复提交为幂等空操作，
        不会重复计数；不同结论覆盖旧值并递增 revision。
        """
        now_ms = int(time.time() * 1000)
        now_iso = _utc_iso(now_ms)

        def write(conn: sqlite3.Connection) -> tuple[Feedback, bool]:
            receipt_row = conn.execute(
                "SELECT 1 FROM prediction_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
            if receipt_row is None:
                raise ReceiptNotFoundError(f"receipt {receipt_id!r} not found")

            row = conn.execute(
                "SELECT actual_abnormal, revision, created_at, updated_at "
                "FROM inspection_feedback WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO inspection_feedback "
                    "(receipt_id, actual_abnormal, revision, "
                    " created_at_ms, updated_at_ms, created_at, updated_at) "
                    "VALUES (?, ?, 1, ?, ?, ?, ?)",
                    (
                        receipt_id,
                        int(actual_abnormal),
                        now_ms,
                        now_ms,
                        now_iso,
                        now_iso,
                    ),
                )
                return (
                    Feedback(
                        receipt_id=receipt_id,
                        actual_abnormal=actual_abnormal,
                        revision=1,
                        created_at=now_iso,
                        updated_at=now_iso,
                    ),
                    False,
                )

            if bool(row["actual_abnormal"]) == actual_abnormal:
                # 重复提交相同结论：不重复计数，也不改写 revision / 时间
                return (
                    Feedback(
                        receipt_id=receipt_id,
                        actual_abnormal=actual_abnormal,
                        revision=row["revision"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    ),
                    False,
                )

            revision = int(row["revision"]) + 1
            conn.execute(
                "UPDATE inspection_feedback SET "
                "actual_abnormal = ?, revision = ?, updated_at_ms = ?, updated_at = ? "
                "WHERE receipt_id = ?",
                (int(actual_abnormal), revision, now_ms, now_iso, receipt_id),
            )
            return (
                Feedback(
                    receipt_id=receipt_id,
                    actual_abnormal=actual_abnormal,
                    revision=revision,
                    created_at=row["created_at"],
                    updated_at=now_iso,
                ),
                True,
            )

        return self._submit(write)

    # ---------- 误报 / 漏报报告 ----------
    def report(self) -> Report:
        """按“每个回执最新结论”统计误报、漏报与已反馈样本数。

        单条 JOIN 聚合语句 + 一个读事务：并发更正时只可能读到提交前或后的
        一致快照，不会混用新旧记录。同一盘的不同预测各有回执，分别计样。
        """
        try:
            self._ensure_ready()
            conn = self._connect()
            try:
                with self._read_tx(conn):
                    row = conn.execute(
                        "SELECT "
                        "  COUNT(*) AS feedback_samples, "
                        "  COALESCE(SUM(CASE WHEN p.risk_score >= ? "
                        "                    AND f.actual_abnormal = 0 "
                        "                    THEN 1 ELSE 0 END), 0) "
                        "    AS false_positives, "
                        "  COALESCE(SUM(CASE WHEN p.risk_score < ? "
                        "                    AND f.actual_abnormal = 1 "
                        "                    THEN 1 ELSE 0 END), 0) "
                        "    AS false_negatives "
                        "FROM inspection_feedback f "
                        "JOIN prediction_receipts p ON p.receipt_id = f.receipt_id",
                        (HIGH_RISK_THRESHOLD, HIGH_RISK_THRESHOLD),
                    ).fetchone()
            finally:
                conn.close()
        except ReceiptStoreUnavailable:
            raise
        except (sqlite3.Error, OSError) as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise ReceiptStoreUnavailable(
                f"report query failed: {self._last_error}"
            ) from exc

        return Report(
            high_risk_threshold=HIGH_RISK_THRESHOLD,
            feedback_samples=int(row["feedback_samples"]),
            false_positives=int(row["false_positives"]),
            false_negatives=int(row["false_negatives"]),
        )
