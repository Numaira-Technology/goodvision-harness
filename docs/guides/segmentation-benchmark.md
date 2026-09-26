# 本机分割模型对比

独立比较 RF-DETR-Seg、YOLOv8-seg、YOLO26-seg 的人物轮廓和速度。
不启动现有服务，不改线上配置，不需要其他分割模型提供结果。

当前搭建机器是 M4 Pro / 48GB，普通终端可使用 Apple MPS。
若迁移到已租好的 4090，先看[云端准备包说明](../../tools/vast_benchmark/README.md)。
现有 Mac 离线工具继续保留。当前 4090 正式验收请使用
[固定 main 的完整显示评测](segmentation-pipeline-benchmark.md)，无需重新部署 SAM。
本页的标注准确率和离线耗时与完整显示描边率是不同指标，不能混为最终排名。

## 开始测试

在仓库根目录的普通终端执行。已搭好的这台电脑可跳过安装。

```bash
# 新机器/环境需要安装时使用；Python 3.12，优先使用 uv。
bash tools/setup_seg_benchmark.sh

# 不下载权重，只检查依赖和设备。
bash tools/run_seg_benchmark.sh doctor

# 先测试三个 Nano，1 次重复。首次运行自动下载官方权重，需要联网。
bash tools/run_seg_benchmark.sh run \
  --models rfdetr-n,yolov8-n,yolo26-n --device mps --repeats 1

# 全部 12 个配置，每项预热 3 帧、正式重复 3 次。
bash tools/run_seg_benchmark.sh run --models all --device mps
```

每个模型在独立子进程中顺序运行。终端会显示当前模型、日志位置及完成状态；
子进程失败后继续下一个模型，最后返回非零状态。模型日志中的错误不会被当作零延迟。
每次运行建立新目录，结束时打印 `report.html` 的绝对路径，双击即可浏览。

默认样本集 `outputs/seg-benchmark/dataset/` 已准备好：从本地已有的
12 段公开来源样片各取前 3 秒、1 FPS，共 36 帧，保留来源与校验值。
这些 360×202 样片用于首次跑通和预览，**不是完整场景验收集**。
其中“事件负样本”仍可能有人，不能当作无人帧。尚无人工轮廓标注。

## 换自己的样本

固定样本只需准备一次，全部模型复用同样 JPEG 和时间戳。可重复传入 `--video`：

```bash
bash tools/run_seg_benchmark.sh prepare \
  --video '/绝对路径/单人.mp4' --video '/绝对路径/多人遮挡.mp4' \
  --max-videos 2 --seconds 30 --sample-fps 3 --max-side 960 \
  --output outputs/seg-benchmark/my-dataset

bash tools/run_seg_benchmark.sh run \
  --dataset outputs/seg-benchmark/my-dataset --device mps
```

也可以传 `--video-dir` 扫描指定目录，或 `--source-manifest` 读取包含
`clips: [{path, sha256?, ...}]` 的已有清单。相对视频路径以清单所在目录为基准。
无输入参数时，使用本机已有的 `tmp/eval/benchmark-12clips-360x202/manifest.json`；
清单不存在会要求指定视频，不会连接摄像头。输出目录已存在时拒绝覆盖。

建议正式集覆盖单人、多人、小人物、遮挡、快运动、暗光和真正无人场景，
调参集与正式测试集按视频/摄像头分开。默认最多取 6 段，需要更多时显式调整 `--max-videos`。
导出的时间是视频帧号除以容器 FPS；可变帧率素材应先转为固定帧率，
不能用这一时间近似宣称真实采集到显示的延迟。

## 标注和 70% 判定

打开样本目录的 `annotate.html`，可直接用浏览器打开文件：

1. 沿一个人物的可见边界逐点点击，选择“完成人物”。
2. 同一人物被遮挡成多个区域时，画下一块并“追加到上一人物”。
3. 标完本帧**所有可见人物**后“审核本帧”。无人帧也要审核。
4. “导出 annotations.json”，记录实际下载路径；下次可导入继续编辑。

页面不会自动把标注写回样本目录。导出后，通过参数使用该文件：

```bash
bash tools/run_seg_benchmark.sh run --device mps \
  --annotations '/绝对路径/annotations.json' \
  --budget-ms 500 --iou 0.5 --target 0.70
```

`--iou 0.5` 是轮廓匹配门槛，`--budget-ms 500` 是**模拟结果截止时间**，
不是强制让每次推理在 500ms 内结束。有效描边率：

```text
在截止时间内完成且 mask IoU ≥ 门槛的人物实例数 / 全部标注人物实例数
```

每帧执行一对一匹配，漏检、过期结果不计成功，重复轮廓计入误检。
另报所有 GT 人物的 IoU、边界 F1（3 像素容差；未匹配记 0）、
误检实例数和无人帧误报比例。人物多边形使用评估 JPEG 的像素坐标，可多块组成一人；
当前简易标注器不支持显式扣除多边形内部孔洞，不标遮挡后不可见部分。

没标注时仍能测速度、看描边，质量字段为 `null`。
只标部分帧时显示这部分的诊断分数，但**只有全部测试帧审核完且存在 GT 人物，
才给出达标/不达标判定**。空场景不算 100% 达标。多次重复不增加独立样本量。
这些分数是当前样本集成绩，不能自动代表所有摄像头或线上稳定性。

## 模型和公平性

| ID | 模型 | 默认内部方形尺寸 |
| --- | --- | --- |
| `rfdetr-n/s/m/l` | RF-DETR-Seg Nano/Small/Medium/Large | 312 / 384 / 432 / 504 |
| `yolov8-n/s/m/l` | YOLOv8 n/s/m/l-seg | 640 |
| `yolo26-n/s/m/l` | YOLO26 n/s/m/l-seg | 640 |

默认比较原生推荐尺寸的部署配置，不宣称相同内部输入尺寸。
同一批原始 JPEG、相同置信度（默认 0.25）、batch=1，FP32 请求，
不启用外部混合精度、TensorRT、Core ML、模型编译或测试增强。
模型内部的预处理、原生后处理和实现差异保留并计入时间。

`--size` 可覆盖单次运行中所选模型的尺寸：YOLO 必须为 32 的倍数，
RF-DETR Nano 为 12、其余为 24 的倍数。共同尺寸可用 384 或 576，
也可在单个模型上试多个尺寸，每次使用新的输出目录，先调参再锁定测试集。

```bash
bash tools/run_seg_benchmark.sh run \
  --models rfdetr-s,yolov8-s,yolo26-s --device mps --size 384
```

类别通过模型实际类别名识别：YOLO 的 person 为 0，RF-DETR 的 COCO ID 不能照搬。
GPU 前后同步计时，输出 mask 必须对应输入图像尺寸。权重和依赖首次下载不计入稳态时间。
模型顺序使用固定 seed 打乱；正式比较建议关闭其他重负载程序、连接电源并记录机器状态。

## 输出含义

每次运行位于 `outputs/seg-benchmark/run-*/`：

| 文件 | 内容 |
| --- | --- |
| `report.html` | 汇总表和同源帧各模型描边预览 |
| `summary.csv` / `summary.json` | 速度、质量、截止时间曲线、失败原因 |
| `settings.json` / `annotations.json` | 参数、执行顺序、数据校验值、冻结的标注 |
| `dependencies.txt` | 本次运行环境的全部依赖版本 |
| `<模型>/result.json` | 设备、软件版本、权重 SHA256、加载/预热耗时和结果 |
| `<模型>/frames.jsonl` | 每帧、每次重复的计时与评分 |
| `<模型>/*.jpg` / `*.npz` | 第一轮描边图和完整布尔 mask |
| `<模型>/run.log` | 下载、模型警告、进度和错误 |

- `pipeline_ms`：**内存 BGR 图像 → 预处理 → 模型推理 → CPU mask → 轮廓**。
  不含读盘/JPEG 解码、评分、输出文件、网络、跟踪和屏幕显示。
- `simulated_ready_ms`：把上述每帧耗时放到样本时间线上，模拟单路 FIFO 累积排队。
  每段视频、每次重复分别清空队列。这是模拟值，不是实测 Harness 端到端延迟。
- `effective_coverage`：当前截止时间下的有效描边率。
  JSON 另给 50/100/200/500/1000/1500/4000ms 等预算的覆盖曲线。
- 内存是进程 RSS 和设备分配字节的采样值；MPS 采样可能漏掉帧内峰值，
  不等于整机/整卡峰值。

本工具交付的是**离线模型选型测试台**，没有改动现有 Harness 实时推理后端。
不提供跨帧人物 ID、实时播放覆盖率、双路资源调度或 60 分钟稳定性验收。
稀疏采样的描边率也不代表完整 15/30 FPS 视频的覆盖率。

## 故障处理

- `doctor` 中 MPS 不可用：在 macOS 普通终端执行。受限执行环境可能看不到 Metal。
- 明确使用 `--device mps`，可避免环境异常时 `auto` 选择 CPU。
- 某个模型出现 MPS 算子错误：检查对应 `run.log`。可用 `--device cpu` 单独诊断；
  不要开启 `PYTORCH_ENABLE_MPS_FALLBACK=1` 后把混合执行结果记为 GPU 成绩。
- 下载失败：修复网络后重新运行失败的模型。已成功下载的权重缓存于 `.models/seg-benchmark/`。
- 依赖安装进入 `.venv-seg-benchmark/`，与服务的 `.venv/` 分离。
  核心依赖版本固定于 `tools/seg_benchmark/requirements.txt`，实际传递依赖记录在每次结果中。

来源：[RF-DETR](https://github.com/roboflow/rf-detr)、
[YOLOv8](https://docs.ultralytics.com/models/yolov8/)、
[YOLO26](https://docs.ultralytics.com/models/yolo26/)。
