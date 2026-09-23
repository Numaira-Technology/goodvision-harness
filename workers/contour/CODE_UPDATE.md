# GoodVision Contour 轻量更新 {release}

用于已经安装 20260920 完整包的 Windows GPU 机器。仅包含运行代码，
不含模型、CUDA、Python 依赖或完整镜像，无需联网和额外安装 Python。
更新会复用原镜像层；新增磁盘占用主要是不到 1 MB 的代码层。
模型保持原样，服务重启时仍需等待模型加载和 GPU 预热。

## 安装

1. 在主程序停止两路分析，等待队列结束。
2. 将此 ZIP 复制到 GPU 机，完整解压到一个新文件夹。
3. 打开解压后的文件夹，双击 UPDATE.cmd。也可在该文件夹打开 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\update_contour.ps1
```

脚本自动定位现有 goodvision-contour 服务，校验代码和基础镜像，离线生成代码层，
保留原来的 .env、模型目录、密钥、端口、GPU 和两路配置。仍使用
http://192.168.20.93:8093，主程序无需重新配置。
如发现会话运行中，脚本会退出，停止分析后重新运行即可。
出现 Update ready 后安装完成。脚本不自动开始视频分析。
刷新节点控制台并输入现有密钥，可看到显存读数。

## 本次改进

- 模型预热完成后回收空闲显存缓存。
- 最后一路结束、超时或取消后，在下一次空闲维护周期（约 5 秒）回收缓存。
- 另一条视频流仍在运行、或后台 GPU 请求尚未结束时，不进行回收。
- 模型继续常驻显存。显示张量实际占用、空闲缓存和整卡剩余显存。
- 新增经现有密钥鉴权的 GET /api/gpu-memory，便于远程诊断。

## 检查与回退

只检查、不更新：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\update_contour.ps1 -CheckOnly
```

更新后就绪检查失败会自动恢复旧版本。需要手动回退时，先停止分析，然后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\update_contour.ps1 -Rollback
```

只保留此版本所需的旧镜像回退标签；它们与新镜像共享大层，不复制模型或依赖。
不要删除模型目录、Docker 数据卷或回退标签。重复运行不会叠加安装。
原完整包的重新安装会恢复原代码；日常更新使用此轻量包。
更新记录位于原 workers/contour/.contour-updates/ 目录。
