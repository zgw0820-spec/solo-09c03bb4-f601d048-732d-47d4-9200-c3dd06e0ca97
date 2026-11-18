# 磁盘 SMART 风险推理服务

基于固定权重 NumPy 逻辑回归模型的 FastAPI 推理服务，面向机房按 SMART 测值安排磁盘巡检的场景。
并发请求在服务端合成为**真正的一次矩阵推理**，并带严格的期限、排队与故障隔离语义。
成功的预测会写入 SQLite **可追溯回执**，运维凭回执提交该盘实际是否异常，
据此核对风险模型的**误报 / 漏报**。

## 功能与语义

### 推理

| 需求 | 实现 |
| --- | --- |
| 输入 | 磁盘号 `disk_no`、温度 `temperature`、重映射扇区数 `reallocated_sectors`、待映射扇区数 `pending_sectors`、独立总期限 `deadline_ms` |
| 输出 | `risk_score`（0~1）、`batch_id`（每次矩阵推理递增的推理批次号）、`receipt_id`（可追溯回执号） |
| 模型 | 固定权重的 NumPy 逻辑回归（标准化 + 一次矩阵乘 + sigmoid），权重随代码固定、可复现 |
| 合批 | 单批最多 **8** 个；组批窗口不超过 **40ms**；每批只做一次矩阵推理 |
| 排队 | 最多 **32** 个名额，满员直接 `503` |
| 独立期限 | 每个请求从接入时刻起算总期限；排队超时 `504` 并腾出名额；成批后单请求到期只判自己失败，**不拖累同批其余请求**；过期结果绝不报成功 |
| 断连 | 客户端断连后排队项被移除、腾出名额（20ms 级轮询感知），不返回成功 |
| 故障隔离 | 模型处理失败使该批整批 `500`，后续批次不受影响；该批**不发放回执** |
| 文档 | Swagger UI：`/docs`；OpenAPI：`/openapi.json` |

### 回执与巡检反馈（SQLite）

| 需求 | 实现 |
| --- | --- |
| 可追溯回执 | 只有**请求期限内成功（200）**的预测才登记回执，响应中的 `receipt_id` 即凭据；`504 / 500 / 503 / 499` 一律不回执、不落库 |
| 回执查询 | `GET /receipts/{receipt_id}` 返回该次预测的盘号、分数、批次号与当前生效的反馈；不存在返回 `404` |
| 提交巡检结论 | `POST /receipts/{receipt_id}/feedback`，body `{"actual_abnormal": true/false}`；回执不存在 `404` |
| 同盘多预测 | 每次成功预测生成独立回执，**同一盘的不同预测分别计样** |
| 重复反馈 | 同一回执同一结论重复提交为幂等空操作，**不重复计数**，`revision` 不变 |
| 更正 | 不同结论覆盖旧值，`revision` 加 1，`corrected=true`；报告**只认最新结论** |
| 误报 / 漏报 | `GET /feedback/report` 返回 `high_risk_threshold`（固定 **0.5**）、`feedback_samples`（已反馈样本数）、`false_positives`（误报：分数 ≥0.5 但实际正常）、`false_negatives`（漏报：分数 <0.5 但实际异常） |
| 并发一致性 | 报告为**单条 JOIN 聚合语句 + 单个 SQLite 读事务**，只读每个回执的最新结论；与并发更正不混用新旧记录，只会读到更正前或更正后的完整快照 |
| 故障时推理 | 回执库不可用时线上推理照常按原期限给分，响应 `receipt_id` 为 `null`（不发放无法填报的假回执）；反馈查询/写入返回 `503` 并带 `Retry-After`，回执库恢复后自动继续发回执，无需重启 |
| 重启有效 | SQLite 使用 WAL + `synchronous=FULL`，回执与反馈落盘；Compose 中数据库位于命名卷 `receipt-data`，容器重启/重建后已发回执与反馈仍有效 |

## 目录结构

```
app/
  config.py    # 环境变量配置（合批参数、模拟延迟、故障注入、回执库）
  model.py     # 固定权重逻辑回归模型 + 一次性故障注入
  batcher.py   # 异步合批引擎（队列、窗口、期限、故障隔离）
  receipts.py  # SQLite 回执/反馈存储（写线程组提交、读事务快照）
  main.py      # FastAPI 路由与异常映射
tests/
  selftest.py                  # 11 项合批/期限/故障关键自测
  test_env_failure.py          # FAIL_BATCH_NO 环境变量故障路径验证
  test_receipts.py             # HTTP 层：回执、反馈、误报/漏报、并发一致性
  test_receipts_store.py       # 存储层：0.5 边界、幂等/更正、并发、重开库
  test_restart_persistence.py  # 重启后回执与反馈仍有效（自管服务进程）
Dockerfile
docker-compose.yml
```

## 用 Docker Compose 启动

```bash
docker compose up --build
# 服务监听 http://localhost:8000 ，Swagger: http://localhost:8000/docs
```

`docker-compose.yml` 已配置模拟推理延迟与一次故障：

- `MODEL_LATENCY_S=0.12`：每次矩阵推理模拟 120ms 延迟（在线程池 sleep，不卡事件循环）；
- `FAIL_BATCH_NO=3`：第 3 次矩阵推理注入一次故障（该批 500、不发回执，之后自动恢复）；置 `0` 关闭；
- `RECEIPT_DB_PATH=/srv/data/receipts.sqlite3`，`/srv/data` 挂载命名卷 `receipt-data` 持久化回执与反馈。

重置演示数据：`docker compose down -v`（删除卷，回执与反馈一并清空）。

## 本地直接运行（无 Docker 时）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
MAX_BATCH_SIZE=8 BATCH_WINDOW_MS=40 MAX_QUEUE=32 \
MODEL_LATENCY_S=0.12 FAIL_BATCH_NO=0 \
RECEIPT_DB_PATH=data/receipts.sqlite3 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 调用示例

```bash
# 1) 预测：拿到分数与回执号
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"disk_no": 128, "temperature": 46.5,
       "reallocated_sectors": 24, "pending_sectors": 6,
       "deadline_ms": 2000}'
# 200 -> {"disk_no":128,"risk_score":0.37...,"batch_id":7,"receipt_id":"9f2c..."}

# 2) 巡检后凭回执提交实际结论（true=实际异常）
curl -X POST http://localhost:8000/receipts/9f2c.../feedback \
  -H 'Content-Type: application/json' -d '{"actual_abnormal": true}'
# 200 -> {"receipt_id":"9f2c...","actual_abnormal":true,"revision":1,"corrected":false,...}

# 3) 更正结论（只认最新一次）
curl -X POST http://localhost:8000/receipts/9f2c.../feedback \
  -H 'Content-Type: application/json' -d '{"actual_abnormal": false}'
# 200 -> {"revision":2,"corrected":true,...}

# 4) 查询回执与当前结论
curl http://localhost:8000/receipts/9f2c...

# 5) 误报 / 漏报 / 已反馈样本数
curl http://localhost:8000/feedback/report
# {"high_risk_threshold":0.5,"feedback_samples":1,"false_positives":1,"false_negatives":0}
```

## 自测

```bash
# 主实例（FAIL_BATCH_NO=0）
BASE_URL=http://127.0.0.1:8000 python tests/selftest.py
BASE_URL=http://127.0.0.1:8000 python tests/test_receipts.py   # 需全新回执库
python tests/test_receipts_store.py                          # 存储层，无需服务
# 故障注入实例（FAIL_BATCH_NO=2，需全新实例）
BASE_URL=http://127.0.0.1:8001 python tests/test_env_failure.py
# 重启持久化（脚本自行启动/重启服务，使用临时库，需 8010 端口空闲）
python tests/test_restart_persistence.py
```

- `selftest.py`：单请求可复现、12 并发合批（≤8/批、一次矩阵推理）、40ms 窗口、
  独立期限互不拖累、排队上限 32 的 503、排队超时 504 腾名额、断连腾名额、
  故障批 500 且后续批恢复 200。
- `test_receipts.py`：仅成功预测发回执、504/500 不回执不落库、同盘多预测分别计样、
  重复反馈幂等、更正只认最新结论、0.5 界线判定、未知回执 404、
  并发更正与报告查询的一致性、并发预测回执数=成功数。
- `test_receipts_store.py`：直接测存储层——0.5 边界（0.5 算高风险、0.4999 不算）、
  同盘多回执计样、幂等/更正、未知回执、并发写入与报告、重开数据库文件后数据一致。
- `test_restart_persistence.py`：重启前后回执/反馈/报告一致，重启后可继续更正。

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MAX_BATCH_SIZE` | 8 | 单批最大请求数 |
| `BATCH_WINDOW_MS` | 40 | 组批窗口上限（毫秒） |
| `MAX_QUEUE` | 32 | 排队名额上限 |
| `MODEL_LATENCY_S` | 0.12 | 每次矩阵推理的模拟延迟（秒） |
| `FAIL_BATCH_NO` | 0 | 第 N 次推理注入一次故障，0 为关闭 |
| `RECEIPT_DB_PATH` | `data/receipts.sqlite3` | SQLite 回执库路径（Docker 镜像内 `/srv/data/receipts.sqlite3`） |
| `RECEIPT_DB_BUSY_TIMEOUT_MS` | 1000 | SQLite 等锁超时（毫秒） |
| `RECEIPT_WRITE_TIMEOUT_S` | 1.0 | 单次回执/反馈写入等待上限（秒），超时按回执库不可用处理 |

## 接口与状态码约定

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/predict` | 推理；返回分数、批次号、回执号（回执库不可用时为 `null`） |
| `GET` | `/receipts/{receipt_id}` | 查询回执与当前反馈 |
| `POST` | `/receipts/{receipt_id}/feedback` | 提交/更正巡检结论 |
| `GET` | `/feedback/report` | 误报数、漏报数、已反馈样本数 |
| `GET` | `/health` / `/stats` | 健康检查 / 指标（含回执库状态与记录数） |
| `POST` | `/debug/arm-failure` | 【仅演示/自测】武装一次模型故障 |

- `200` 成功；`404` 回执不存在；`503` 排队满员或回执库不可用；`504` 请求独立期限超时；
- `500` 该批模型处理失败；`499` 客户端断连（非标准，仅服务端语义）。
