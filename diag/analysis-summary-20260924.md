# 卡顿/雪花排查总结 — 2026-09-24

目标:手机发 WebRTC → 电脑 Harness 接收,排查 **full annotation** 下为什么卡顿 + 出雪花。

## 测试设置
- 手机浏览器 → WHIP(VP8)→ MediaMTX(本机,HTTPS 自签)→ Harness 拉 WHEP。
- Harness:本机 Mac(**无本地 GPU**),venv 运行。
- SAM:远程 Vast **RTX 4090** 实例(`sam31_stream`,H264 上传,512px),经 SSH 隧道到 `localhost:8092`。
- 事件模型:OpenRouter(远程 VLM)。features = fight/smoking/fall/weapon(**未含 stranger**,因本机未装 InsightFace)。

## 关键结果(满标注 + SAM 真跑起来,见 `harness-SAM-full-20260924.log` / `receive-metrics-SAM-full-20260924.txt`)
分析真在跑:`src_frames=1453`,SAM `submitted=108 succeeded=21 transport_req=23`,远程 SAM 单次延迟 `~2.6s`。

| 指标 | 值 | 结论 |
| --- | --- | --- |
| RTP 丢包 lostΔ | **0** | 无雪花来源(雪花=UDP 丢包→残帧) |
| 事件循环延迟 loop-lag p95 | **0 ms** | 收帧循环**没被 CPU 饿死** |
| 队列丢帧 qdrop | **0** | 分析没堵住收帧 |
| 解码 fps | ~29.6 / 30 | 收帧健康 |
| **DECODE-STALL**(gap>200ms) | **1→5 递增** | 唯一退化信号 |

## 结论
1. **这套设置(远程 SAM + 手机→Mac 局域网)复现不出雪花/卡顿。** 收帧全程健康。
2. **原因:SAM 重计算在远程 GPU,不占本机 CPU** → aiortc 收帧循环不饿死 → 不丢包不雪花。**远程 SAM 架构本身规避了"本机 CPU 抢占→雪花"这条路径。**
3. 唯一递增的 **DECODE-STALL** 最可能是**手机推流不稳**:MediaMTX 全程警告 `received a non-starting fragment`(弱 WiFi / 浏览器 VP8 编码碎片),与 CPU 无关(loop-lag 恒为 0)。
4. 远程 SAM 往返 **~2.6s**:影响标注/显示的新鲜度(滞后),但不造成收帧 stall。

## 生产雪花/卡顿的可能来源(待会上讨论)
- 若生产**也用远程 SAM** → 雪花来自**网络丢包**(与本次 DECODE-STALL 同源);卡顿可能来自**远程 SAM ~2.6s 往返**。
- 若生产**用本地 GPU 跑 SAM** → 才会本机 CPU 抢占 → 雪花。**本次无 GPU 的 Mac 复现不了这条**,需在有本地 GPU 的机器上验(或用 `LOAD=on` 模拟 CPU 满载)。

## 排查过程中修掉/发现的问题(都在今天的改动里)
1. **service 只认 `HARNESS_STREAM_URL` 环境变量,不认 `--stream-url`** → 之前一直连到 .env 的机器人地址、0 帧。
2. **编码:aiortc 只支持 VP8/H264,VP9 不行;浏览器 H264 aiortc 拆不了包** → 必须手机选 **VP8**。
3. **手机摄像头需 HTTPS 安全上下文** → MediaMTX 开自签证书;harness 用 `SSL_CERT_FILE` 信任。
4. **`start_sam.py` 的 `offer_was_unavailable()` bug**:Vast 把 offer 不可用包成 `error=invalid_args`(真因 `no_such_ask` 在 msg 里),导致不跳下一个 offer 就放弃。已修(本地单测通过)。
5. **SSH 卡住**:Vast 账户没登记 `id_ed25519_numaira.pub`,新实例不带 key → 需 `vastai attach ssh` 临时救,或把公钥登记进 Vast 账户(治本)。
6. **启动分析的正确端点是 `/v1/sessions/start`(带 features payload),不是 `/v1/analysis/start`**(后者是 fleet 协调路径,单流会 409)。`HARNESS_AUTOSTART` 只连预览、不启动分析。
7. stranger 需要本机 InsightFace(未装)→ 起会话时 503;去掉 stranger 即可。

## 保存的文件(diag/logs/)
- `harness-SAM-full-20260924.log` — 满标注 + SAM 真跑的完整 harness 日志
- `receive-metrics-SAM-full-20260924.txt` — 解析后的收帧指标表
- `harness-SAM-409-20260924.log` — 用错端点(analysis/start 409)的日志
- `mediamtx-snapshot-20260924.log` — MediaMTX 日志(含 non-starting fragment 警告)
- `webrtc_full_20260924-*.log` — 各轮 run_diag_test 记录

---
## 追加(2026-09-27):现场抓到雪花 + 成因坐实
- iPad WebRTC(VP8)+ 远程 SAM 满标注,面板出现**雪花**(花屏截图见 `snow-screenshot-*.webp`)。
- **同一刻 harness 侧**:RTP 丢包 0、loop-lag 0、队列丢帧 0(仅 DECODE-STALL gap>200ms=2)。
- **MediaMTX**:phonecam 路径 `non-starting fragment` 处理错误累积(12→31)。
- 结论:**雪花来自 iPad→MediaMTX 的 WiFi 上行丢包/碎片**(上游损坏),harness↔MediaMTX(localhost)无丢包,接收端 CPU 健康。**雪花是网络成因,不是接收电脑/CPU。**
- 治理方向:改善发送端网络(近 AP/有线/降码率/密关键帧),或用 TCP 的 RTMP(会变延迟而非花屏)。

---
## 追加(2026-10-03):每路视频的压力归因工具 + 一个解析器 bug

### 解析器 bug(影响上一轮"loop-lag=0"这句话的可信度)
- `diag/parse_receive_log.py` 原来按 key `loop_lag` 取事件循环延迟,但 harness 实际发的字段是 **`stats_loop_lag`**(`stream_diagnostics.py:138`),精确匹配取不到 → `loop p95` 这一列**一直被写死成 0**、`LOOP-LAG` 永不触发。
- **已修**。用修好的解析器重跑上一轮真实日志(`harness-SAM-full-20260924.log`):**max loop p95 = 1ms**。
- 结论:上一轮"接收循环没被 CPU 饿死"的判断**本身成立**;只是之前那一列不可信,现在可信了。

### 新增:每路 × 每环节 的 CPU/GPU/带宽 归因
为回答"**每多加一路视频,新增压力落在 CPU/GPU/带宽的哪个环节**",补了三件:
1. **进程内 `resource_timing`**(`src/goodvision_harness/resource_trace.py`):后台线程每 5s(`RESOURCE_SAMPLE_SECONDS` 可调)记进程 CPU%/RSS/线程数、系统 CPU、以及**按线程名拆到环节的 CPU**(prepare=环节3 编码 / commit=环节6 / display=环节7 / capture=环节1)。每进程=一路,天然按流归因。装 `pip install -e ".[diag]"`(psutil)。
   - 注意:**按环节拆 CPU 仅在 Linux 准确**(线程 id 对齐);macOS 上会全落到 `other`,但进程级 CPU 仍准。生产/fleet 在 Linux,无影响。
2. **`diag/gpu_sampler.sh`**:SSH 进租的 GPU 盒跑 `nvidia-smi` 落 `diag/logs/gpu-*.csv`(harness 看不到远程 GPU,只能看到环节4 的 HTTP 延迟)。
3. **`diag/parse_pipeline_log.py`**:把 `network_receive_timing` / `resource_timing` / `sam_client_http_timing` / `sam_coordinator_timing` / `sam_display_timing` / 显示投递 + GPU CSV 汇总成**每路一块 + 多路并排对比表**,并标出随路数线性增长 vs 饱和的列。
   - 另给 `render_scene_jpeg` 加了 decode/draw/encode 三段计时(进 `sam_display_timing`),把环节7 的 CPU 拆开。

### 用这套工具跑真实日志的早期结论(单路,远程 SAM)
`parse_pipeline_log.py diag/logs/harness-SAM-full-20260924.log`:
- **环节4 远程 SAM 往返 = 绝对大头**:p50 ≈ 2.47s,p95 ≈ 3.18s(与会上 ~2.6s 吻合)。这是 **GPU + 网络** 成本,不是本机 CPU。
- 环节6 commit ≈ 13ms(可忽略)。
- **带宽**:接收(ingest)≈ **98 Mbps**,上行到 SAM 仅 **0.23 Mbps**(帧缩到 512px)——**接收带宽远大于上行**。
- 推断"每多一路"的边际成本:**主机 CPU(解码/编码/绘制)+ 接收带宽 线性叠加;GPU 走共享入场队列是真正的天花板**。要坐实多路 GPU 曲线,需在有 GPU 的机器用上面三件一起跑 1/2/4 路(本轮工具已就绪,未端到端跑)。

### 怎么跑
```bash
pip install -e ".[diag]"                          # 装 psutil
# 1) 各路 harness 正常跑(每进程会自动发 resource_timing)
# 2) 另开一个终端采 GPU:
diag/gpu_sampler.sh root@<vast-host> <ssh-port>   # 落 diag/logs/gpu-*.csv
# 3) 汇总(每路一个日志文件,想比几路就给几个):
python diag/parse_pipeline_log.py s1.log s2.log s4.log --gpu-csv diag/logs/gpu-*.csv
```

---
## 追加(2026-10-03 晚):真跑了 GPU 多路(1/2/4 路,每路独立 lane)

实例:Vast RTX 4090($0.35~0.39/hr),worker `SAM31_LIVE_CAMERAS=4`(4 lane 共享一个模型)。源:循环 `0057...fight.mpeg` 走 RTSP → N 个 harness 进程,**每路指向自己的 lane**(`RICH_LABEL_ENDPOINT=.../v1/cameras/camera-0N/perception/sam_live/sessions`)。**全部跑完已销毁,剩余实例数 0。**

> 更正:多路**是支持的**。harness 的 `PerceptionSessionHTTPClient`(`perception.py:70-80`)会在 endpoint 路径为 `/v1/cameras/camera-0N/...` 时把 session 绑到对应 lane——**不需要改 harness 代码**。第一次 N≥2 失败是我的压测脚本把所有 harness 都指到了单一默认端点(`/v1/video/sessions`,`max_active_sessions:1`),撞在同一个 native session 上才 `native_session_failed`。改成每路独立 lane 后 1/2/4 路都正常并发。

### 结果曲线(每路独立 lane)
| N | 环节4 SAM p50 | 环节4 p95 | GPU util(mean/max) | GPU 显存 | 每进程 CPU | 系统 CPU |
|---|---|---|---|---|---|---|
| 1 | **252 ms** | 274 ms | 63% / 79% | **7.8 GB** | ~41% | 51% |
| 2 | ~420 ms | ~2.5 s | (采样窗口错位,不可靠) | ~高 | ~42% | 45% |
| 4 | **2708 ms** | **~8–10 s** | 39% / 85% | **21.1 GB** | ~41% | 81% |

(环节6 commit 全程 7–19ms;解码 24fps 满、loop-lag 0、丢包 0、qdrop 0——接收侧始终健康。N=2 的 GPU util 因为前一次 sweep 被 kill 导致采样窗口错位,数值不可信;N=1/N=4 干净。)

### 瓶颈结论(多路 SAM pipeline)
1. **延迟随 lane 数急剧恶化**:环节4 SAM 往返 p50 **252ms(1路)→ ~420ms(2路)→ 2708ms(4路)**,p95 **274ms → 2.5s → ~9s**。4 路约 **10×**。根因:worker `max_gpu_operations:1`——**4 条 lane 的 GPU op 串行排队**,互相等。GPU 平均 util 在 4 路反而只有 39%(max 85%),说明不是算力打满,而是**串行 + 每请求开销(传输/调度)让 GPU 在 op 之间有空隙**——即瓶颈是**串行准入 + 单请求往返开销**,不是纯算力吞吐。
2. **显存是硬上限**:≈ **5 GB/lane**(1路 7.8GB → 4路 21.1GB)。24GB 的 4090 **最多 ~4 路**(第 5 路会爆显存)。
3. **主机 CPU 是线性成本**:每路 ~0.4 核(此机),系统 CPU 51%→81%。本地(SAM off)2 路就能把 Mac 顶到 ~100%——真实部署里主机 CPU 也会随路数线性涨。
4. **接收侧全程健康**:解码满帧、loop-lag/丢包/qdrop 全 0。**雪花与此 pipeline 无关**(发送端 WiFi 上行丢包)。

**一句话**:多路 SAM 的瓶颈 = **GPU 侧的串行准入(`max_gpu_operations:1`)→ 延迟随路数线性/超线性恶化 + 显存 ~5GB/lane 的硬上限(24GB≈4路)**;主机 CPU 是线性成本但不是先撞到的那堵墙;单路时 GPU 有余量,瓶颈只是那一次 ~0.25s 往返。

### 复现
```bash
# worker 起 N lane:
SAM31_LIVE_CAMERAS=4 docker compose -f compose.yaml -f compose.sam31-remote.yaml \
  -f diag/compose.sam-tunnel.yaml -f diag/compose.sam-multistream.yaml up --build sam-remote
# 采 GPU + 跑 N 路(每路自动指到 camera-0N lane):
diag/gpu_sampler.sh root@<host> <port>            # 或在 run 内联
STREAMS=4 VIDEO=<file> SAM=on SECS=90 GPU_CSV=diag/logs/gpu-*.csv diag/run_multistream_diag.sh
```
注:提升多路吞吐的方向 = 放开 worker `max_gpu_operations`(让多 lane 真并行,受显存约束)或多卡(`cloud_pool.py` 每卡一进程);`codex/twelve-stream-runtime` 的 `FairGPUAdmission` / `cloud_pool` 就是干这个的。

---
## 追加(2026-10-03 深夜):修复 + 验证 —— `RICH_LABEL_LATEST_ONLY`(新鲜度背压)

**诊断**:4 路时环节4 p50=2708ms、p95~9s,但 GPU 只有 39% util——瓶颈不是算力,是 harness **把帧喂得比 SAM 排得快**,`_pending` 积压到 ~36s(`queue_oldest_seconds:36`),结果越算越旧;单飞往返被积压拖长,GPU 在 op 之间空转。

**修复**(纯 harness 侧,免费):给流水线协调器加 `latest_only` 背压——落后时**丢最旧的未处理帧**,只留最新 ~`pending_capacity` 帧。每个 SAM batch 独立(fresh predictor epoch),丢旧帧不破坏时序。开关 `RICH_LABEL_LATEST_ONLY`(默认关=保留离线基准的"全量不丢";开=直播/多路要新鲜)。改动:`config.py`(字段+env)、`perception.py`(透传)、`sam_tracking_pipeline.py`(`submit` 丢最旧);单测覆盖丢弃/保留/env,流水线全套 22 项绿。

**GPU 实测 A/B(N=4,同机 RTX 4090)**:

| 指标 | 修复前 | 修复后(latest_only) | 改善 |
|---|---|---|---|
| 环节4 SAM p50 | 2708 ms | **659 ms** | ~4× |
| 环节4 SAM p95 | ~8–10 s | **2.65 s** | ~3–4× |
| 90s 内结果数/路 | ~27 | **84** | ~3× |
| GPU util mean | 39% | **69%**(max 100%) | 喂满了 |
| GPU 显存 | 21 GB | 23.4 GB | 4路近满 |

**结论**:这个**免费的客户端侧修复把多路延迟降了 ~4×、吞吐提了 ~3×**,顺带把 GPU 利用率从 39% 拉到 69%——因为积压本身就在饿着 GPU。剩下的 **2.65s p95(4路)+ 23.4/24GB 显存** 是**结构性的 GPU 串行(`max_gpu_operations:1`)+ 显存硬上限**,要再进一步只能靠多卡/更大显存(硬件/钱),不是代码能修的。

**怎么用**:直播/多路部署给 harness 设 `RICH_LABEL_LATEST_ONLY=true`;离线基准保持默认(关)以保证全量。
