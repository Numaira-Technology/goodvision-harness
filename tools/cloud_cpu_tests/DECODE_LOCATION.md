# 视频解码位置对照

用同一段约 56 秒的测试视频、同一台 4090、同一个云端 Harness/SAM 服务，比较两个浏览器位置：

- **Mac 条件**：Mac 浏览器从本地读取并解码 MP4，经 SSH 隧道上传帧给云端 Harness，结果回到 Mac 浏览器呈现。
- **云端条件**：云端无头浏览器从云端文件读取并解码同一 MP4，在同一主机上传帧给 Harness，也在云端浏览器呈现。

这是真正搬动浏览器解码与上传位置的诊断。只把 MP4 文件放到云端、仍由 Mac 浏览器播放，改变的是文件下载位置，不是解码位置。云端条件**没有把预览视频传回 Mac**；因此它不能单独证明完整产品体验会同样提速，也不能与此前 Mac 系统录屏测试直接合并。

先用保存的同一视频内容；摄像头直播会引入画面变化，待视频对照确认有价值后再测试。原测试片为 HEVC，云端 Playwright Chromium 未必能解码，所以先在 Mac 转成一份 H.264，再把**同一个转码文件**复制到云端。两侧必须使用同一个 SHA256；转码后的结果不能直接与旧 HEVC 批次作严格数值对照。该实验使用真实 SAM 与 VLM，不会自动租卡。默认 1 秒显示缓冲，至少各跑两轮，建议顺序 Mac → 云端 → 云端 → Mac，避免启动顺序偏差。

在 Mac 生成这份输入（现有工作区已生成 `outputs/cloud-decode/input-h264.mp4`）：

```bash
mkdir -p outputs/cloud-decode
ffmpeg -hide_banner -loglevel error -y -i video_samples/sample_video.mp4 \
  -map 0:v:0 -an -c:v libx264 -preset veryfast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart outputs/cloud-decode/input-h264.mp4
shasum -a 256 outputs/cloud-decode/input-h264.mp4
```

## 云端准备

使用包含当前代码的测试包；本地可用 `tools/export_seg_benchmark.py --meeting-tests --skip-offline-dataset --pipeline-video video_samples/sample_video.mp4 --output dist/cloud-decode.tar.gz` 重新打包后上传。另将 `outputs/cloud-decode/input-h264.mp4` 上传到云端包的 `video_samples/decode_input.mp4`，不要在两侧各自转码。云端安装并启动 SAM 和 Harness 的方式沿用 [测试 1、2 说明](README.md)，但启动 Harness 时加上转码文件路径：

```bash
.venv-harness-benchmark/bin/python -m tools.cloud_cpu_tests.server \
  --allow-face-download --video video_samples/decode_input.mp4
```

只运行一个测试 Harness 服务；不要和旧服务占用同一端口。云端浏览器还需要 Node.js、npm 和 Playwright Chromium：

```bash
node --version
npm install --prefix .cache/seg-pipeline --save-exact playwright@1.62.1
.cache/seg-pipeline/node_modules/.bin/playwright install --with-deps chromium
```

Mac 保持 [README](README.md) 中的 SSH 端口转发。两端必须有哈希相同的视频文件；脚本会核对云端服务报告的 SHA256，错误时拒绝运行。云端需安装 Node.js 18 或更新版本。

新实例准备完成后，可在 Mac 上用参数化脚本运行完整的 Mac → 云端 → 云端 → Mac 对照（先保持 18094 端口隧道可用）：

```bash
bash tools/cloud_cpu_tests/run_decode_comparison.sh 'INSTANCE_IP' 'SSH_PORT'
```

可选第 3、4 个参数是 SSH 私钥路径和云端包目录。脚本在两侧固定使用相同的四帧上传窗口、15 秒 ACK 超时和 30 秒片尾排空等待。第一轮若 VLM 请求失败就会停止，避免继续生成无效对照。输出写入新的 `outputs/cloud-decode/<UTC 时间>/` 目录，不会覆盖历史数据。

## 运行

在 Mac 包根目录运行第一轮：

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.decode_location run \
  --location mac --video outputs/cloud-decode/input-h264.mp4 \
  --output outputs/cloud-decode/mac-1 --delay 1
```

在**云端**包根目录运行第一轮：

```bash
.venv-harness-benchmark/bin/python -m tools.cloud_cpu_tests.decode_location run \
  --location cloud --video video_samples/decode_input.mp4 \
  --output outputs/cloud-decode/cloud-1 --delay 1
```

再按相反顺序运行云端第二轮和 Mac 第二轮，分别把输出目录改成 `cloud-2`、`mac-2`。每轮必须等上一轮浏览器和云端队列排空后再开始。将云端的 `cloud-1`、`cloud-2` 输出目录复制回 Mac 相同路径，然后在 Mac 汇总：

```bash
.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.decode_location compare \
  --mac outputs/cloud-decode/mac-1 --cloud outputs/cloud-decode/cloud-1 \
  --mac outputs/cloud-decode/mac-2 --cloud outputs/cloud-decode/cloud-2 \
  --output outputs/cloud-decode/comparison.md
```

每轮 `summary.json` 保存视频哈希、测试脚本哈希、浏览器版本、描边率、呈现 FPS、VLM 请求状态、上传配置、云端收帧到结果及采集到呈现 P95，以及两端 CPU 采样。只要 VLM 请求没有全部成功，本轮就会标记为无效；汇总还会拒绝视频、缓冲、服务器、模型、GPU、浏览器分析配置或测试脚本版本不一致的轮次，并要求两侧各至少两轮。汇总先展示逐轮结果和上传积压，再展示描述性平均值。浏览器版本和主机 CPU 仍可能不同；报告中会明示，不能只凭一次差异判定稳定提速。
