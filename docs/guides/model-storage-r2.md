# 事件 benchmark 视觉模型固定存储（Cloudflare R2）

> 2026-11-03 调研 + 方案。目标：把事件识别 benchmark（fight/smoking/fall/weapon）用到的**本地推理视觉模型**放进一个固定的 Cloudflare R2 bucket，按 revision 锁定统一管理，避免每次租 GPU 都从 HuggingFace 反复下载（慢、被限速、还可能花钱）。
>
> 本文能离线查证的都已查证并给出结论；需要凭据/账号才能做的（真正上传、跑 live `vastai cloud copy`）列在文末 [待办与需要用户提供的信息](#待办与需要用户提供的信息)，没有卡住其余工作。

## 范围：哪些模型进 R2

只把**在 GPU 上本地跑权重**的模型放进 R2。纯 API 模型没有权重，不进。

| 模型 | 用途（benchmark 类别） | HF repo | 钉死 revision | 代码出处 | 是否进 R2 |
|---|---|---|---|---|---|
| VideoJev-4B | decision（system-one 决策，native） | `ZhuYuhan/VideoJev-4B` | `1c1a253bf90dfa66176124d25a48ffb791d517d5` | `src/goodvision_harness/videojev_core.py:7-8` | **是** |
| CLIP ViT-L/14 | threshold（neural / CLIP 分数） | `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` | `src/goodvision_harness/neural_service.py:18-19` | **是** |
| SigLIP2 base | threshold（计划接入） | `google/siglip2-base-patch16-224` | **未钉死**（本分支代码里还没有，需补） | 任务指定，暂无代码引用 | 是（待补 revision） |
| d3-nano | decision（本地 decision3 运行时） | `vllm-sr/d3-nano` | `462efc366229ea268cf4eed3738b3fc4415e4817` | `src/goodvision_harness/decision3_runtime.py:16-17` | 是（本地跑才需要） |
| Qwen3.5-35B-A3B FP8 | 自部署 Qwen 基线（Vast） | `Qwen/Qwen3.5-35B-A3B-FP8` | `9d1823d2dee688a6b25e77009dc727688c44936e` | `tools/vast_vlm/fp8_profile.py:10-11` | 可选，见下 |
| Qwen3.5-35B-A3B BF16 | 自部署 Qwen 基线（110GiB+ 显存） | `Qwen/Qwen3.5-35B-A3B` | `59d61f3ce65a6d9863b86d2e96597125219dc754` | `docs/guides/vast-self-hosted-qwen35.md` | 可选，见下 |

**不进 R2：**

- **language 类**（`qwen/qwen3-vl-8b-instruct`、`google/gemini-2.5-flash`）走 OpenRouter API，没有本地权重。出处 `tools/event_benchmark/run_eval.py:41`、`tools/event_benchmark/SPEC.md:72`。
- **Luna decisions**（`openai/gpt-6-luna-decisions`）同样是 API。
- 用户说 **Qwen 各尺寸权重已在团队 NAS/checkpoint**。NAS 上的 Qwen 语言权重不必镜像进 R2——它们在 benchmark 里走 API。只有**自部署 Qwen3.5-35B（FP8/BF16）基线**真的在 Vast 上加载本地权重；是否把这两个（FP8 ≈35GB、BF16 ≈70GB）也放进 R2，取决于是否要摆脱对 NAS 可达性 / HF 反复下载的依赖——**建议先只放 FP8**（体积一半，是当前 4090/48GB 入口用的精度），BF16 按需再说。需用户确认（见待办）。

## R2 目录布局与命名

单 bucket（bucket 前 10 万个免费，不需要多 bucket）：

```
goodvision-models/                         # bucket
└── models/
    └── <hf_org>/<hf_name>/<full_40char_commit_sha>/<HF 原始相对路径>
```

实例：

```
models/ZhuYuhan/VideoJev-4B/1c1a253bf90dfa66176124d25a48ffb791d517d5/
    config.json  model-00001-of-....safetensors  tokenizer.json  ...
models/openai/clip-vit-large-patch14/32bd64288804d66eefd0ccbe215aa642df71cc41/
    config.json  model.safetensors  preprocessor_config.json  ...
models/vllm-sr/d3-nano/462efc366229ea268cf4eed3738b3fc4415e4817/
models/Qwen/Qwen3.5-35B-A3B-FP8/9d1823d2dee688a6b25e77009dc727688c44936e/
```

规则：

- **key = `models/<org>/<name>/<完整 40 位 commit SHA>/<文件原始路径>`。** 永远用完整 commit SHA，不用 tag / 分支名（tag 会移动，SHA 不会）。R2 里的 SHA 必须和代码里的 `REVISION` 常量**逐字相同**。
- **按 HF 原始文件树 1:1 镜像**（含子目录），这样下游工具可以把 `local_dir` 直接指向该 revision 目录，和 `snapshot_download(..., revision=..., local_dir=...)` 的落盘结构一致（`tools/vast_vlm/runtime.py:195`、`tools/videojev_probe/__main__.py:60`）。
- 每个 revision 目录放一个 **`MANIFEST.json`**：文件名 → sha256 + 字节数，和 runner 现有的 `verified.json` 校验逻辑对齐，拉取后可校验完整性。
- 每个 revision 目录放一个 **`_ADDED.json`**（上传人、时间、来源 HF URL），记录来源，便于季度清理时判断。
- **只上传权重相关文件**：`.safetensors / .json / .txt / .model / .jinja`，和 `runtime.py` 的 `allow_patterns` 一致；不要上传 `.bin` 双份、`.msgpack`、`*.h5`、README 图片等无关大文件。

## 管理规则（selective、定期、不要 abuse）

直接回应约束“定期管理、别无限堆 revision”：

1. **选择性**：只有上表“是”的本地权重进 R2。API 模型、NAS 上够用且 benchmark 不本地加载的权重，一律不进。
2. **一个 (repo, revision) 一个目录，immutable。** 代码 bump revision 时**新增**目录，不覆盖旧的。
3. **保留策略：每个模型最多留 2 个 revision**（当前 + 上一个用于回滚）。新 revision 合并进 main 且验证通过后，删掉更早的。**代码/文档里没有任何地方引用的 revision，一律删。**
4. **不要开 bucket 的 object versioning。** revision 目录本身已经是显式版本管理；再开 R2 versioning 会悄悄堆积 noncurrent 版本，而这些版本**仍然计入计费存储**（官方计费按存储字节算）。如果出于别的原因必须开，就配一条 lifecycle 规则 expire noncurrent 版本。*(注：R2 object-versioning 官方文档页当天返回 404，上传后请在控制台确认 noncurrent 版本是否计费，再决定。)*
5. **Lifecycle：** abort incomplete multipart uploads > 7 天（避免上传中断残留的分片字节被计费）。
6. **容量预算：** 稳态目标 **< 60 GB**（视觉集 ×2 revision + 可选一个 Qwen FP8）。预算是治理手段不是省钱——即便 100 GB 也只有约 $1.5/月（见下）。纪律在于“不留死 revision”，不是省那点钱。
7. **谁能写：** 两个 token——
   - **只读 token**（Object Read only，限定本 bucket）：发给 runner / GPU 实例，用于拉取。
   - **读写 token**（Object Read & Write，限定本 bucket）：只给 1–2 名维护者，用于上传。
   - 两个 token 都**绝不进 repo**，走密码管理器分发，定期轮换。
8. **季度复查：** 列 bucket → 逐个 revision 目录对照代码 `REVISION` 常量 → 删孤儿 → 校验 MANIFEST → 看总容量。

## Vast AI CLI 能否直接下模型（已验证）

本地装了 `vastai` 1.10.0（`pip install vastai`）离线核对了命令面；**没有** Vast 账号，所以没跑真实传输——下面是命令可用性与签名的结论，live 传输列入待办。

**结论：`vastai` CLI 没有“直接从 HF 模型 repo 下载”的命令。** 全量命令列表里不存在这类子命令。可用的相关命令只有：

- `vastai cloud copy --src SRC --dst DST --instance INSTANCE_ID --connection CONNECTION_ID --transfer "Cloud To Instance"`
  在**已注册的云连接**和实例之间传文件。支持 `--dry-run`、`--size-only`、`--ignore-existing`、`--schedule`。**仅 Docker 实例可用**（VM/KVM 不行）。
- `vastai show connections` 列出账号下的云连接（拿 CONNECTION_ID）。**CLI 没有创建连接的命令**——云连接要在 Vast 网页控制台（Account → Cloud Connections）里建。
- `vastai copy [instance:]path [instance:]path` 在实例/本地之间 rsync 式拷贝（走 Vast relay，不涉及云存储）。

**对我们的意义——R2 是 S3 兼容的**，所以有两条路，推荐第 2 条：

1. **`vastai cloud copy` 走 R2**：先在 Vast 网页控制台把 R2 注册成一个 S3 兼容云连接（填 R2 的 S3 endpoint `https://<account_id>.r2.cloudflarestorage.com` + Access Key/Secret），`vastai show connections` 拿到 `CONNECTION_ID`，然后：
   ```bash
   vastai cloud copy \
     --src /goodvision-models/models/ZhuYuhan/VideoJev-4B/1c1a253bf90dfa66176124d25a48ffb791d517d5/ \
     --dst /workspace/models/videojev/ \
     --instance <INSTANCE_ID> --connection <CONNECTION_ID> \
     --transfer "Cloud To Instance"
   ```
2. **在实例内直接拉（推荐，最少依赖、不限实例类型）**：用 `rclone` / `aws s3` / `boto3` 打 R2 的 S3 endpoint，或用 HF 原路 `hf download`。这和 repo 现有 `snapshot_download(..., revision=...)` 行为一致。
   ```bash
   # R2（egress 免费），rclone remote 名为 r2：
   rclone copy r2:goodvision-models/models/ZhuYuhan/VideoJev-4B/1c1a253bf90dfa66176124d25a48ffb791d517d5/ \
     /workspace/models/videojev/ --transfers 8 --s3-no-check-bucket
   # 或从 HF 原路（会被限速、偶发不稳，正是要规避的）：
   hf download ZhuYuhan/VideoJev-4B --revision 1c1a253bf90dfa66176124d25a48ffb791d517d5 \
     --local-dir /workspace/models/videojev/
   ```

> HF 的 cloud sync 只支持 HF 的 **S3 Storage Buckets**，不暴露 model/dataset repo——所以“用 vastai 从 HF 模型页直接同步”这条路不通。我们自己的 R2 走 S3 兼容则没问题。

## 上传（seed）命令

需要读写凭据后执行。先从 HF 按钉死 revision 下到本地，再推到 R2。

```bash
# 1) 按钉死 revision 从 HF 下到本地（用 repo 自带下载器，revision 已内置）
python -m tools.videojev_probe download --cache-dir /tmp/dl        # VideoJev @ 1c1a253...
#   CLIP/SigLIP2/d3-nano 可直接 hf download <repo> --revision <sha> --local-dir ...

# 2) 推到 R2（rclone，remote 名 r2；或用 aws s3 --endpoint-url）
rclone copy /tmp/dl/<snapshot>/ \
  r2:goodvision-models/models/ZhuYuhan/VideoJev-4B/1c1a253bf90dfa66176124d25a48ffb791d517d5/ \
  --s3-no-check-bucket

# aws-cli 等价写法
aws s3 sync /tmp/dl/<snapshot>/ \
  s3://goodvision-models/models/ZhuYuhan/VideoJev-4B/1c1a253bf90dfa66176124d25a48ffb791d517d5/ \
  --endpoint-url https://<account_id>.r2.cloudflarestorage.com
```

`rclone` 的 R2 remote（`~/.config/rclone/rclone.conf`，**凭据不进 repo**）：

```ini
[r2]
type = s3
provider = Cloudflare
access_key_id = <R2_ACCESS_KEY_ID>
secret_access_key = <R2_SECRET_ACCESS_KEY>
endpoint = https://<account_id>.r2.cloudflarestorage.com
acl = private
```

## R2 价格与成本估算

官方价格页（`developers.cloudflare.com/r2/pricing/`，页面标注 **Last updated 2026-10-30**，当天查证）：

| 项 | Standard | Infrequent Access (IA) |
|---|---|---|
| 存储 | **$0.015 / GB-月** | $0.01 / GB-月 |
| Class A 操作（写/列，PUT/LIST/Copy） | $4.50 / 百万 | $9.00 / 百万 |
| Class B 操作（读，GET/HEAD） | $0.36 / 百万 | $0.90 / 百万 |
| 数据取回（retrieval） | 无 | $0.01 / GB |
| **出口 egress（传到公网）** | **免费** | **免费** |

- **免费额度（每月，仅 Standard）**：存储 10 GB-月、Class A 100 万、Class B 1000 万。
- Bucket：每账号前 10 万个免费，之后 $0.10/个/月。
- IA 有 30 天最短存储期（提前删也按 30 天计），**无免费额度**；我们是“写少、反复读、长期留”，用 **Standard**，不要用 IA。
- 计费细节：用量向上取整到下一个计费单位；存储按 30 天内每日峰值平均；DeleteObject/DeleteBucket/AbortMultipartUpload 免费；认证失败的请求不计费。

**成本估算**（按 Standard $0.015/GB-月；模型体积为估计值，上传后以实际为准）：

| 场景 | 体积（估） | 存储/月 |
|---|---|---|
| 视觉集单 revision（VideoJev≈8 + CLIP≈1.7 + SigLIP2≈0.4 + d3-nano≈2） | ≈ 12 GB | ≈ $0.18（基本落在 10GB 免费内） |
| 视觉集 ×2 revision + 一个 Qwen FP8（≈35） | ≈ 59 GB | ≈ $0.89 |
| 宽松上限 | ≈ 100 GB | ≈ $1.50 |

- **操作成本可忽略**：上传一次全部分片约数百次 PUT（远低于 100 万免费）；每次租机拉取几十~几百次 GET，就算 100 次租机 ×500 GET = 5 万，也远低于 1000 万免费。
- **egress 免费是关键**：相比每次新租机从 HF 重拉（分钟级占用**付费 GPU** + HF 限速/偶发失败），或付 Vast 磁盘费，R2 的 S3 API + 免费 egress 让再水合又快又免费。

**对比 Vast 自带 storage：** Vast 的实例/卷磁盘按**宿主报价**计费（各 offer 不同，常见约 $0.05–0.20/GB-月），且**只在实例/卷存活期间**收费——实例一销毁就没了，于是下次又得重下。network volume 能持久但仍是宿主定价且绑定地区。真正的隐性成本不是磁盘费，而是**每次新租机从 HF 重下**：占用付费 GPU 的等待时间 + HF 不稳（正是本任务要解决的痛点）。R2 把权重钉在一处、egress 免费、S3 协议直连，按月几乎零成本就消除了这个重复下载问题。

## 待办与需要用户提供的信息

能直接做的已做（查价、验证 Vast CLI、定布局与规则、写本文）。下面这些需要账号/凭据，**真正上传模型到 R2 被这些 block**：

1. **确认 Cloudflare 账号**：团队是否已有带 R2 的 Cloudflare 账号？提供 **account ID**。
2. **Bucket**：是否已存在（名字）？没有就建 `goodvision-models`（Standard，关 versioning）。
3. **R2 S3 凭据**（走密码管理器 / 密钥管理，**不要贴进 repo 或聊天**）：
   - 读写 API token 的 **Access Key ID + Secret Access Key**（限定该 bucket），用于上传。
   - S3 endpoint：`https://<account_id>.r2.cloudflarestorage.com`。
   - （可选）再要一个只读 token 发给 runner 拉取。
4. **确认模型清单**：
   - 自部署 **Qwen3.5-35B FP8** 要不要进 R2（建议：要；BF16 暂不）？
   - **SigLIP2 的确切 revision**（`google/siglip2-base-patch16-224` 当前本分支代码里还没引用，需钉死一个 commit SHA）。
5. **Vast（可选）**：是否要在 Vast 网页控制台把 R2 注册成云连接以用 `vastai cloud copy`？若只用 rclone/hf 在实例内拉取（推荐），这步可省。

拿到 1–3 后即可按上面的 seed 命令上传并建 MANIFEST；4–5 是清单/路径确认。
