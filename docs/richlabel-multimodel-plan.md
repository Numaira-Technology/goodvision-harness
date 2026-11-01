# 实现 Plan:RichLabelWorker 多模型选项(YOLO / RF-DETR,控制台可选)

## 1. 背景与目标
SAM 又准又慢(实测多路 4 路 p95 ~2.6s+、GPU 串行)。production 更想要**快**的非大模型检测器。Frank 的要求:
- 控制台能**下拉选** RichLabelWorker 当前用哪个模型(SAM 之外再加 YOLO、RF-DETR)。
- 按 Haotian 2026/09/29 的测试结果,每款留一版最好的。
- 目标:同样的"描边"效果,换更快的检测器,production 更顺。

## 2. 关键事实(已走查代码确认)
**输出契约**:SAM 现在每帧每目标输出一个 `RichLabelObject`(`src/goodvision_harness/rich_labeling.py:79`):
- 必填:`track_id`、`label`、`bbox`(归一化);可选:`confidence`、`attributes.mask_contours`(描边)、`observation_state`。
- **框必填,mask 可选** → 只出框的检测器就满足契约,`scene_delivery.render_scene_jpeg` 无描边时自动画矩形框。
- **`track_id` 必须由 worker 返回**,harness 不生成,只做一帧内唯一性校验 + 跨帧按 track_id 关联(`rich_labeling.py:223-225, 1017-1029`)。

**已有可复用的(几乎是全部骨架)**:
- `workers/paddle_rich_label/app.py`(1377 行):检测 → **`IoUTracker`**(`:123`,检测器无关的 IoU 跟踪器,`_next_id` 分配 track_id、超时淘汰)→ 按 `/v1/labels` 返回。
- harness 侧:`RichLabelHTTPClient`(`rich_labeling.py:346`,POST `/v1/labels`)、跨帧身份关联、`RICH_LABEL_BACKEND ∈ {paddle,vlm,sam,sam31,sam31_stream}`(`config.py:322`)。
- 控制台:`dashboard.html` 已有 `inference-mode`/`tracking-mode` 两个 `<select>` + `deployment_settings.py` 持久化模式 —— 新下拉照抄。

**结论**:跟踪器、契约、harness 客户端、关联、控制台下拉模式**全现成**。真正要新写的只有 **YOLO/RF-DETR 的推理封装**。

## 3. 选型(来自测试结果,取"0.5s 达标 + 描边率高")
| 控制台选项 | 权重 | 速度档 | 两轮描边率 | 角色 |
|---|---|---|---|---|
| SAM(现有) | sam31_stream | ~2.6s+ | 最高、出 mask | 最准 |
| RF-DETR | **rfdetr-l** | 0.5s | 81.8/81.2 | DETR 最优 |
| YOLO·快 | **yolo26-s** | 0.5s | 79.8/81.7 | 最快 |
| YOLO·准 | **yolov8-m** | 1s | 83.9/87.4 | YOLO 最准 |

(yolov8-n、yolo26-n 淘汰:各档未双轮达标。)

## 4. 实现(两块)

### A. Worker 侧 —— 新增 detector worker(照抄 paddle)
新目录 `workers/detector_rich_label/`(或扩 paddle worker):
1. **骨架照抄** `paddle_rich_label/app.py`:FastAPI、`/v1/labels`、健康检查、响应格式、**直接复用 `IoUTracker` + `Detection`/`Track`**。
2. **换推理引擎**:加一个 `MODEL` env(或请求参数)选 `yolov8m` / `yolo26s` / `rfdetrl`,加载对应权重:
   - YOLO:ultralytics(`model.predict` → 框+类别+conf)。
   - RF-DETR:官方推理 → 框+类别+conf。
   - 输出统一转成 `Detection(label, bbox_normalized)`(只保留 person 类,`label="person"`),喂给 `IoUTracker.update()` → `Track`(带 track_id)→ 按契约返回。`confidence` 顺带带上。
3. **权重**:每款一版放进镜像/挂载;worker 启动时按 `MODEL` 加载一个(像 SAM 的 `SAM31_*` env)。
4. Dockerfile:ultralytics + rfdetr 依赖(GPU 版)。

> 不做:mask 分割头(契约可选,先出框)。

### B. Harness / 控制台侧 —— 加"检测模型"下拉
1. **config/deployment_settings**:在 `DeploymentSettings`(`deployment_settings.py:36`)加一个字段,例如 `rich_label_model`(取值 `sam | rfdetr-l | yolo26-s | yolov8-m`),save/load + 校验。
2. **选择 → 落到 rich_labeling 配置**:开会话时按所选模型设置 `backend` 与 `endpoint`:
   - `sam` → 现有 sam31_stream 路径(worker 8092)。
   - 其余三者 → `backend=paddle`(复用 `RichLabelHTTPClient`/`/v1/labels`),`endpoint` 指向 detector worker;把所选 `MODEL` 透传给 worker(请求参数或各自端点)。
   - 复用现有 `RICH_LABEL_BACKEND`/`RICH_LABEL_ENDPOINT` 机制,不改 harness 核心管道。
3. **dashboard.html**:照 `inference-mode`/`tracking-mode` 加一个 `<select id="richlabel-model">`(4 个 option),`fleet_console.js` 里读/写该设置(POST 到已有的设置保存端点),`model-chip` 顺带显示当前模型。

## 5. 改动文件清单
- 新增:`workers/detector_rich_label/`(app.py 照抄 paddle + 推理封装、Dockerfile、requirements、权重配置)。
- 改:`src/goodvision_harness/deployment_settings.py`(加 `rich_label_model` 字段 + 校验/持久化)、`service.py`(开会话时按所选模型设 backend/endpoint)、`config.py`(如需放宽 backend 校验/加映射)、`dashboard.html` + `fleet_console.js`(下拉 + 读写设置)。
- 复用不动:`rich_labeling.py`(契约/客户端/关联)、`IoUTracker`、`scene_delivery`(框 fallback)。

## 6. 范围 / 非目标
- 做:4 个模型下拉可选、detector worker(出框 + track_id)、控制台切换。
- 不做(本轮):mask 描边(检测器出框即可)、多模型同时跑(单选切换)、模型对比评测(已有 Haotian 的结果)。

## 7. 验证
1. **契约单测**:detector worker 的 `/v1/labels` 响应能被 `RichLabelObject.from_dict` 接受(track_id/label/bbox 齐全、一帧内 track_id 唯一)。
2. **跟踪连续性**:同一人跨帧 track_id 稳定(IoUTracker 单测,复用 paddle 现有测法)。
3. **端到端**:控制台切到 yolo26-s / rfdetr-l / yolov8-m / sam,各跑一路,面板看到框 + 跟踪稳定;切换生效。
4. **延迟对照**:每个模型跑一路量 source→显示延迟,核对与测试结果一致(0.5s/1s 档)。

## 8. 工作量 / 分工
- 大头是 **detector worker 的推理封装**(YOLO + RF-DETR 各一段,跟踪/契约/骨架全复用)—— 1 人 ~1-2 天。
- 控制台下拉 + 设置透传 —— 小,半天。
- Frank 说"跑一天 Codex 就出来"基本靠谱:骨架可抄、契约清晰、跟踪器现成。

## 9. 开工前要 Frank 拍板的点
1. YOLO 两版都上(yolo26-s 快 + yolov8-m 准)还是只留一版?(建议都上,成本几乎一样)
2. 权重从哪来(谁给 .pt / checkpoint)?
3. detector worker 跑在哪(独立 GPU 服务器,像 SAM 那样租/部署)?
