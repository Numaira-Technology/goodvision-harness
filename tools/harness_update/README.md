# GoodVision 主程序离线更新

更新包包含打包时工作区中受版本管理的完整主程序。具体版本、源码提交与文件校验值以包内 `manifest.json` 为准；当前源码不能继续冒用已发布的 cloud-health7 名称。

连续升级会按已知底版的镜像层来源、镜像发布标记及每路两处安装路径的程序 SHA256 识别已发布版本，兼容目标机重新构建后不同的镜像 ID。校验通过后才构建、替换服务。

Windows PowerShell 5.1 会逐个检查已知底版镜像，避免多镜像数组混淆；确实没有可用底版时仍停止升级。请将 ZIP 解压到新的文件夹，再使用新目录中的入口。

1. 在原控制台停止全部分析，等待收尾完成，并关闭自动开始分析。
2. 解压更新 ZIP，无需覆盖原安装目录或移动模型。
3. 标准 Windows 安装（包括 Lemon 的 `goodvision-harness-deploy-harness-1` 和 `harness-02-1`）：双击 `CHECK.cmd` 检查，通过后双击 `UPDATE.cmd` 安装。仅开发机的 `goodvision-live-latest` 和 `goodvision-contour-camera-02` 使用 `CHECK-8080.cmd` / `UPDATE-8080.cmd`；这两个入口按精确容器名选择，并非所有使用 8080 端口的安装都适用。
4. 看到 `Update ready` 后刷新原控制台，连接视频并手动开始分析。线路 C 可选择千问 30B。

更新使用已安装镜像的依赖和模型，无需下载。包含完整受版本管理的应用代码、浏览器资源和许可证，不含个人数据、访问密钥、测试视频或模型权重。

支持原 Windows 20260919 安装、Lemon 已核对镜像 ID 的 diagnostics2 安装、scene-results 20260919、本机 diagnostics1、single-video1、cloud-health1，以及从已验证底版连续升级的 cloud-health2–7。请保留原底版镜像（更新器自动保留的 goodvision-harness-update-base 镜像），供离线核验和运行检查器。更新前校验镜像来源、发布文件和包内文件，并校验原 Windows 安装的启动器元数据；未知、文件不匹配或混合版本不会修改。名字包含 `-before-` 的历史备份容器不参与更新。多个安装可运行 `update_harness.ps1 -Project 项目名` 指定。

如需精确选择其他已有容器，可使用 `update_harness.ps1 -Containers "主容器名,第二路容器名"`；加 `-CheckOnly` 只检查。所有指定容器都必须存在、属于同一部署且版本相同，否则不会修改任何容器。

保留容器名、端口、访问密钥、模型和数据卷、视频地址、播放缓冲、线路选择及原安装启动入口。此次会启用独立 VLM 与持续事件处理，并把播放采集目标设为 15fps；其他环境配置保留。更新需要短暂重启，完成后分析保持停止。本包没有回退功能。

`update-state` 仅保存安装状态和容器标识，不含密钥。只更新所选 GoodVision 主程序，不清理其他 Docker 项目。仅部署 Contour 的 GPU 服务器无需安装本包。

这是现有安装的主程序升级包，不能用于空白电脑全新安装。不包含 Docker 基础镜像、模型权重或 Vast 控制器镜像；不会租用服务器、重启现有 Vast 租赁容器或启动分析。Vast 控制器本身仍使用原有已更新镜像。

## 维护者打包

在仓库根目录执行，明确选择新的发布名：

```powershell
python tools/harness_update/package_update.py --release <新的发布名>
```

打包成功后才将该包的应用校验清单记入 `published_sources.json`。相同发布名只允许相同应用内容；内容变化必须使用新名称，不能覆盖旧记录。该文件用于识别客户已安装的历史版本，不是可随意清理的临时输出。发布时同时提交更新后的清单；ZIP 与校验文件保存在忽略目录 `dist/`。
