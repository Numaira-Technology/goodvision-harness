# 交付与升级：检查边界

[验证索引](README.md) · [更新工具说明](../../tools/harness_update/README.md) · [发布构建](../guides/release.md)

## 连续升级与 Windows PowerShell 5.1

2026-09-21 的 health6/health7 修复解决两类安装识别问题：目标机重建导致镜像 ID 不在静态列表；PowerShell 5.1 将多个镜像的反序列化 JSON 包成嵌套数组，导致 RootFS 比对混合多个镜像。

现有更新器以已知底版运行检查器，按镜像自身发布标记核对已发布来源和安装内容；逐个处理镜像数组，保留 Windows 安装记录、同部署同版本与分析停止检查。不执行尚未核验的更新镜像，不通过放宽文件完整性检查接受未知来源。

验证已覆盖不同构建 ID 的连续升级、多底版且匹配项在第二位、不同入口，以及未知发布标记、错底版、文件篡改和多路混合版本拒绝。实际 Docker 验证只执行 `-CheckOnly`，没有因此构建或重启业务容器。health7 应用 payload 与 health6 相同；**目标客户机器的实际升级结果仍需现场确认**，本地验证不能代替它。

回归入口：[升级链路](../../tests/test_harness_update_chain.py)、[更新包与原生入口](../../tests/test_harness_code_update.py)。当轮包及校验结果位于 `dist/goodvision-harness-update-20260921-cloud-health7.zip` 及同名校验文件；不把这一历史文件名当作以后发布的固定版本。

打包工具现在要求显式指定发布名，拒绝同名版本对应不同应用内容，并在包和校验文件生成成功后才记录历史清单。回归见[发布清单保护](../../tests/test_harness_update_package.py)；该规则防止后续打包覆盖已安装版本的校验依据。

## 对客名称与交付审计

公开界面、状态、结果、SSE、Webhook 和错误采用中性产品名称；内部模型协议、真实业务 ID、普通人名、第三方版权及许可证不能被盲目替换。容器配置中的 `driver: nvidia` 是必要设备选择器，按审计器的精确例外保留。

源码扫描和镜像/离线交付包检查是不同范围。2026-09-18 的源码检查曾通过，但没有重新审计所有既有镜像，也没有在线服务或真实 GPU 验收；该历史通过状态不能沿用到新包。

复跑当前源码检查：

```powershell
.venv-local-gpu/Scripts/python.exe tools/audit_customer_bundle.py . --repository
```

回归见[交付审计测试](../../tests/test_customer_bundle_audit.py)。测试进程设置 `PYTHON_DOTENV_DISABLED=1`，避免部署环境覆盖预期；正式交付仍需按[发布指南](../guides/release.md)检查实际输出包。
