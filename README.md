# 磁盘 SMART 风险推理服务

基于固定权重 NumPy 逻辑回归模型的 FastAPI 推理服务，面向机房按 SMART 测值安排磁盘巡检的场景。
并发请求在服务端合成为**真正的一次矩阵推理**，并带严格的期限、排队与故障隔离语义。

## 功能与语义

| 需求 | 实现 |
| --- | --- |
| 输入 | 磁盘号 `disk_no`、温度 `temperature`、重映射扇区数 `reallocated_sectors`、待映射扇区数 `pending_sectors`、独立总期限 `deadline_ms` |
| 输出 | `risk_score`（0~1）与 `batch_id`（每次矩阵推理递增的推理批次号） |
| 模型 | 固定权重的 NumPy 逻辑回归（标准化 + 一次矩阵乘 + sigmoid），权重随代码固定、可复现 |
| 合批 | 单批最多 **8** 个；组批窗口不超过 **40ms**；每批只做一次矩阵推理 |
| 排队 | 最多 **32** 个名额，满员直接 `503` |
| 独立期限 | 每个请求从接入时刻起算总期限；排队超时 `504` 并腾出名额；成批后单请求到期只判自己失败，**不拖累同批其余请求**；过期结果绝不报成功 |
| 断连 | 客户端断连后排队项被移除、腾出名额（20ms 级轮询感知），不返回成功 |
| 故障隔离 | 模型处理失败使该批整批 `500`，后续批次不受影响 |
| 文档 | Swagger UI：`/docs`；OpenAPI：`/openapi.json` |

## 目录结构

```
app/
  config.py    # 环境变量配置（合批参数、模拟延迟、故障注入）
  model.py     # 固定权重逻辑回归模型 + 一次性故障注入
  batcher.py   # 异步合批引擎（队列、窗口、期限、故障隔离）
  main.py      # FastAPI 路由与异常映射
tests/
  selftest.py          # 11 项关键自测
  test_env_failure.py  # FAIL_BATCH_NO 环境变量故障路径验证
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
- `FAIL_BATCH_NO=3`：第 3 次矩阵推理注入一次故障（该批 500，之后自动恢复）；置 `0` 关闭。

也可以调用 `POST /debug/arm-failure` 确定性武装“下一次推理失败一次”，便于演示与自测。

## 本地直接运行（无 Docker 时）

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
MAX_BATCH_SIZE=8 BATCH_WINDOW_MS=40 MAX_QUEUE=32 \
MODEL_LATENCY_S=0.12 FAIL_BATCH_NO=0 \
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 调用示例

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"disk_no": 128, "temperature": 46.5,
       "reallocated_sectors": 24, "pending_sectors": 6,
       "deadline_ms": 2000}'
# 200 -> {"disk_no":128,"risk_score":0.73...,"batch_id":7}
```

## 自测

```bash
# 主实例（FAIL_BATCH_NO=0）
BASE_URL=http://127.0.0.1:8000 python tests/selftest.py
# 故障注入实例（FAIL_BATCH_NO=2）
BASE_URL=http://127.0.0.1:8001 python tests/test_env_failure.py
```

覆盖：单请求可复现、12 并发合批（≤8/批、一次矩阵推理）、40ms 窗口、
独立期限互不拖累、排队上限 32 的 503、排队超时 504 腾名额、断连腾名额、
故障批 500 且后续批恢复 200。

## 配置项（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MAX_BATCH_SIZE` | 8 | 单批最大请求数 |
| `BATCH_WINDOW_MS` | 40 | 组批窗口上限（毫秒） |
| `MAX_QUEUE` | 32 | 排队名额上限 |
| `MODEL_LATENCY_S` | 0.12 | 每次矩阵推理的模拟延迟（秒） |
| `FAIL_BATCH_NO` | 3 | 第 N 次推理注入一次故障，0 为关闭 |

## 状态码约定

- `200` 推理成功；`503` 排队名额已满；`504` 请求独立期限超时；
- `500` 该批模型处理失败；`499` 客户端断连（非标准，仅服务端语义）。
