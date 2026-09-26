# 固定 main 配置的 4090 描边评测

基于 main `ccff0525abcfa51e8cdb9f56e602cfda909314f3`，替换分割模型，保留 Harness
连续会话、排队、浏览器轮廓传播和显示链路。每个模型默认完整播放原视频两轮，
判断每轮描边率是否达到 70%，记录实际耗时。默认不增加缓冲；显式延迟扫描见下文。

## 固定内容

- main 的 `sam_live` 路由实际显示缓冲为 **0 秒**；普通路由的 2.5 秒不适用于此处。
- 实际跟踪采样为 **2 FPS**；播放显示目标为 15 FPS，保留 main 的 JPEG 传输、
  `latest_only` 排队、轮廓传播及路由派发逻辑。原始 `config.yaml` 不作修改。
- VLM 使用 main 已有的 OpenRouter preset（`qwen/qwen3.5-35b-a3b:nitro`），
  保留 InsightFace CPU 分支。仅切换 provider，不修改 main 的 `config.yaml`。
  人脸图库为空，输出目录隔离，禁用外部 webhook。
- 使用 `video_samples/sample_video.mp4` 原始字节，不转码、不截断。
  SHA-256 为 `2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9`。
- 候选 worker 共用源帧校验与轮廓协议、1280 最大输入边长和 16 人上限，
  各模型内部再按自身默认尺寸推理，统一置信度 0.25。
- 各候选模型统一使用 bbox IoU 关联和三次观测确认；这与 SAM 自带跟踪器有差异，
  会记录到报告元数据中。

SAM 历史结果保留在 `docs/validation/evidence/sam-display-latency-20260922.json`。
**不需要 HF Token、SAM3.1 权重，也不需要安装或启动 SAM。** 共享代码目录中的 `sam`
名称用于复用协议，不会加载 SAM 模型。历史测试有不同缓冲档位，不能直接据此计算
本轮加速倍数，也不会将“1.5 秒”强行作为显示缓冲。历史记录中的 VLM 模型为
`qwen/qwen3-vl-30b-a3b-instruct`，与本轮 preset 不同；历史 SAM 数据仅作参考。

## 1. 云端 GPU worker

将准备包上传至已租好的 RTX 4090 Linux x86_64 实例，解压。需要 Python 3.12，
`nvidia-smi` 可用。基础系统依赖与上传示例见包内 `START_HERE.md`。

```bash
cd goodvision-4090-benchmark
bash tools/vast_benchmark/setup.sh seg
.venv-seg-benchmark/bin/python -m tools.seg_pipeline.worker --model yolov8-n
```

安装脚本会注册 12 个默认停止的 `goodvision-seg` supervisor 服务，供
`run_remote.sh` 自动切换。上面的最后一条命令仅用于手工运行单个 worker；
首次启动自动下载该模型公开权重并预热。看到服务就绪后保持终端运行。
手工模式每次只启动一个候选 worker，按 Ctrl+C 停止后更换 `--model`。
本轮不使用 `setup.sh sam` 或旧离线 `run_seg_benchmark.sh run` 作为最终验收入口。

模型名称：

| 系列 | Nano | Small | Medium | Large |
| --- | --- | --- | --- | --- |
| RF-DETR | `rfdetr-n` | `rfdetr-s` | `rfdetr-m` | `rfdetr-l` |
| YOLOv8-seg | `yolov8-n` | `yolov8-s` | `yolov8-m` | `yolov8-l` |
| YOLO26-seg | `yolo26-n` | `yolo26-s` | `yolo26-m` | `yolo26-l` |

## 2. 本机 Harness 与 Chrome

在 Mac 的同一份源码或准备包根目录执行一次：

```bash
bash tools/seg_pipeline/setup_client.sh
```

使用已安装的 Google Chrome 播放原视频，安装脚本需要 Python 3.12、Node.js/npm。
云端负责分割 GPU 推理；Mac 负责 Harness、浏览器、VLM 调用和 CPU 人脸分支。
所有候选保持相同 Mac、SSH 链路和实例，报告记录客户端环境；浏览器呈现延迟包含网络影响。

另开本机终端，使用 Vast 提供的 SSH 地址和端口：

```bash
ssh -N -L 8092:127.0.0.1:8092 -p YOUR_SSH_PORT root@YOUR_HOST
```

在运行测试的本机终端设置 `OPENROUTER_API_KEY`。可用 zsh 安静读入，
不把密钥写入命令历史或打包文件：

```zsh
read -s 'OPENROUTER_API_KEY?OpenRouter API key: '
export OPENROUTER_API_KEY
```

完整链路会真实调用 VLM API，产生对应费用；只在正式测试时执行下面的命令。

## 3. 正式测试

```bash
.venv-harness-benchmark/bin/python -m tools.seg_pipeline \
  --model yolov8-n --endpoint http://127.0.0.1:8092 \
  --repeats 2 --allow-face-download
```

`--allow-face-download` 允许首次下载 InsightFace 权重。已有权重也可用
`--face-model-root /path/to/insightface` 指定。无需 SAM 权重。
之后更换云端 worker 和本机命令中匹配的模型名称即可。
可用 `--display-delay 1.5` 显式测试单个缓冲档位（0..8 秒），不提供采样率或排队策略调参入口；
本机 `.env` 和部署配置不会覆盖冻结参数。报告记录实际覆盖的缓冲值。
原视频已检查，不会在每次测试前额外重跑播放预检查。

程序最后打印 `outputs/seg-pipeline/<model>-<timestamp>/report.html`。
同时保存 `manifest.json`、`summary.json`、各轮截图、源帧与显示计数、模型耗时、
worker 权重校验值及 GPU 信息。完整播放还要求采集帧与呈现帧一致、没有显示丢帧，
且呈现帧率至少保持目标 15 FPS 的 80%；否则轮次无效并以退出码 1 停止。

## 4. 延迟扫描

在 Vast 实例运行过 `setup.sh seg` 后，本机可执行：

```bash
bash tools/vast_benchmark/run_remote.sh all \
  --host YOUR_HOST --port YOUR_SSH_PORT --identity /path/to/private-key \
  --delays 0.5,1,1.5,2
```

模型在云端只启动一次，本地依次测试每档两轮；两轮各自达到 70% 后停止该型号后续档位。
直接使用已有 GPU worker 时，也可运行：

```bash
.venv-harness-benchmark/bin/python -m tools.seg_pipeline.sweep \
  --model yolov8-n --endpoint http://127.0.0.1:8092 \
  --delays 0.5,1,1.5,2 --repeats 2 --allow-face-download
```

只改变 `display.sam_live_delay_seconds`，并将浏览器显示缓冲绑定为同一值。
原始 `config.yaml` 保持不变。采样、传输、模型及其他设置沿用固定配置。
本轮扫描不会覆盖旧报告，自动生成 `outputs/seg-delay-sweep/<批次>/report.html`。
结果只代表所测离散档位；两轮有效但未达到 70% 时继续下一档，无效测试立即停止。
不假设描边率随延迟单调上升，不把“2 秒内未达标”解读为该模型永远无法达标。

## 如何读结果

- **描边率**：`continuous_contour_frames / distinct_presented_frames`，包含启动阶段
  未描边帧，允许 main 的近似补全。严格描边率另列。默认两轮都达到 70% 才算重复达标。
- **模型 P50/P95**：在 GPU worker 内对一次模型适配器调用计时并同步 CUDA，
  包含预处理、推理和掩码还原，不含网络、关联和浏览器显示。
- **呈现 P50/P95**：浏览器采集源帧到显示该帧的时间，包含无描边帧。它不是模型耗时。
- 固定配置下先筛选重复达标的模型，再比较模型与呈现耗时；不达标就如实报告不达标。
  描边率只统计模型已知人物，漏检可能抬高数值，应结合保存的截图查看漏人情况。
  该指标不是人工真值分割准确率。

协议、计数、配置冻结和打包检查已经完成，RTX 4090 实机安装及完整链路也已跑通；
具体模型、推理速度、轮次和 70% 达标结果以保存的实测报告为准。
