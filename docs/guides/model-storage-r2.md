# 事件 benchmark 视觉模型固定存储（Cloudflare R2）

> 2026-11-03 调研 + 方案。目标：把事件识别 benchmark（fight/smoking/fall/weapon）用到的**本地推理视觉模型**放进一个固定的 Cloudflare R2 bucket，按 revision 锁定统一管理，避免每次租 GPU 都从 HuggingFace 反复下载（慢、被限速、还可能花钱）。
>
> 本文能离线查证的都已查证并给出结论；需要凭据/账号才能做的（真正上传、跑 live `vastai cloud copy`）列在文末 [待办与需要用户提供的信息](#待办与需要用户提供的信息)，没有卡住其余工作。

## 上传状态

| 日期 | bucket | 内容 |
|---|---|---|
| 2026-11-03 | `goodvision-models`（account `2c044451…`，Standard） | VideoJev-4B + CLIP 已上传，共 40 对象 / 10.68 GB（各含 `MANIFEST.json` + `_ADDED.json`） |
| 2026-11-04 | 同上 | 新增 SigLIP2（1.54 GB）+ d3-nano（4.46 GB）；bucket 共 4 模型 / 85 对象 / ≈16.67 GB |

账号 `sx2660@nyu.edu` 的 Cloudflare 已确认带 R2。SigLIP2 与 d3-nano 已上传；Qwen FP8 尚未上传（见 [待办](#待办与需要用户提供的信息)）。

> **上传踩坑（务必注意）**：`huggingface_hub` 若装了 `hf_xet`，大权重会走 HF 的 Xet CDN，期间可能报 `CAS Client Error ... error sending request`，导致只传上去几个小文件就中断。上传脚本因此：① `export HF_HUB_DISABLE_XET=1`（或卸载 `hf_xet`）强制走传统 LFS；② 每个文件最多重试 6 次、指数退避；③ 带 `HF_TOKEN`（本仓库 `.env` 已有）解除匿名限速。缺 token 也能跑，但 8GB 会被限速拖得很慢。

## 范围：哪些模型进 R2

只把**在 GPU 上本地跑权重**的模型放进 R2。纯 API 模型没有权重，不进。

| 模型 | 用途（benchmark 类别） | HF repo | 钉死 revision | 代码出处 | 是否进 R2 |
|---|---|---|---|---|---|
| VideoJev-4B | decision（system-one 决策，native） | `ZhuYuhan/VideoJev-4B` | `1c1a253bf90dfa66176124d25a48ffb791d517d5` | `src/goodvision_harness/videojev_core.py:7-8` | **已上传 ✅ (8.96 GB)** |
| CLIP ViT-L/14 | threshold（neural / CLIP 分数） | `openai/clip-vit-large-patch14` | `32bd64288804d66eefd0ccbe215aa642df71cc41` | `src/goodvision_harness/neural_service.py:18-19` | **已上传 ✅ (1.71 GB)** |
| SigLIP2 base | threshold（计划接入） | `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | `tools/model_source.py` `_KNOWN`（暂无服务引用） | **已上传 ✅ (1.54 GB)** |
| d3-nano | decision（本地 decision3 运行时） | `vllm-sr/d3-nano` | `462efc366229ea268cf4eed3738b3fc4415e4817` | `src/goodvision_harness/decision3_runtime.py:16-17` | **已上传 ✅ (4.46 GB)** |
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
- **镜像该 revision 的完整快照**，只丢掉冗余的重复权重格式：当存在 `.safetensors` 时不传 `.bin / .msgpack / .h5 / .pth`。**务必保留自定义代码 `.py` 文件**——VideoJev 用 `trust_remote_code=True` 加载（`src/goodvision_harness/videojev_core.py:97`），缺了 `.py` 会无法 `from_pretrained`。配置/分词器等小文件（`.json / .txt / .model / .jinja`）照常保留。

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

## 在 Vast GPU 上丝滑取用（runner 集成）

**设计原则：核心加载代码不动，只在 runner 预取阶段把数据源从 HF 换成 R2。** 模型服务（`videojev_core` / `neural_service`）仍按 repo id + revision 从 `cache_dir` 的 HF 缓存布局加载（`local_files_only`），我们只是先把缓存填好。无 R2 凭据时自动退化为纯 HF，所以可安全合并、增量迁移。

- **新模块 `tools/model_source.py`**：`ensure_cached(repo, revision, cache_dir)`，顺序 **本地已校验 → R2（有凭据时）→ HF 兜底**；把 `models/<repo>/<rev>/` 按 HF 缓存布局（`models--org--name/snapshots/<rev>/`）填进 `cache_dir`，并按 bucket 的 `MANIFEST.json` 逐文件校验 sha256。也可当 CLI 用。
- **完整性边界**：R2 下载先写临时目录，全部校验通过后才发布 HF snapshot；失败的临时目录会丢弃，避免 HF 兜底复用半成品。固定 revision 不改写 HF `refs/main`。已有 R2 snapshot 若校验失败（文件损坏、manifest 无效或身份不符），会直接报错；检查并移走该 revision 目录后再重试，不把已知损坏的文件交给 HF 缓存快路径。
- **凭据**：从环境读 `R2_ENDPOINT/R2_BUCKET/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY`，controller 经 SSH 带到 GPU（和 `HF_TOKEN` 一套，不进 repo/镜像）。服务启动前 `tools/vast_vlm/runtime.py` 会把 `R2_*` / `HF_TOKEN` / `*API_KEY*` 从服务进程环境抹除。
- **VideoJev**：`videojev_probe download` 已改为走 `ensure_cached`；随后 `run` / `server` 用同一 `--cache-dir` 原样加载。
- **CLIP**：无需改核心——controller 先跑预取 CLI，再照常起 `neural_service`（同一 `--cache-dir`，设 `HF_HUB_OFFLINE=1`）。

```bash
# 预取（R2 优先、HF 兜底）——revision 取自代码里钉死的常量
python -m tools.model_source --model videojev --cache-dir /models/cache
python -m tools.model_source --model clip     --cache-dir /models/cache
# 服务原样从缓存加载
python -m tools.videojev_probe run --cache-dir /models/cache ...
HF_HUB_OFFLINE=1 python -m goodvision_harness.neural_service --cache-dir /models/cache --port 8900
```

端到端：

```
vastai 租机 → SSH + onstart(修权限) → controller 带 R2+HF 凭据
  → model_source 预取(R2 拉 + MANIFEST 校验, ~2min; 本地已有则秒过)
  → 服务 local_files_only 加载 → 跑 event benchmark
重租同款(或挂持久卷): 本地已校验 → 跳过下载
```

预取环境需 `boto3`（已加进 `tools/videojev_probe/requirements.txt`）；缺了也能走 HF 兜底。每个 runner 各取各的模型，VideoJev 与 CLIP 同机或分开都适用。

### 自动租机：通用远端预取（`tools/cloud_prefetch.py`）

harness 的 SAM 风格自动租机脚本（`start_sam.py` 及复用其机器的 `start_detector.py` 等）在 SSH 就绪、起 worker 之前，调一行就把权重从 R2 预取到实例——**凭据和 `HF_TOKEN` 一样走 SSH stdin，不进 argv/镜像**；R2 不可达时远端自动走 HF 兜底。worker 无关、与具体模型解耦。

```python
from cloud_prefetch import prefetch_models   # start_sam 同目录
host, port = wait_for_ssh(vast, instance_id, options)
cache = prefetch_models(
    ssh_options=options, host=host, port=port,
    remote_dir=f"{REMOTE}/models",
    models=[(MODEL, REVISION)],                 # 任意 (repo, 40位 sha)
    r2_env={k: os.environ.get(k) for k in ("R2_ENDPOINT","R2_BUCKET","R2_ACCESS_KEY_ID","R2_SECRET_ACCESS_KEY")},
    hf_token=os.environ.get("HF_TOKEN"),        # 兜底时解限速
)
# 随后用 cache 作为 worker 的 --cache-dir（服务 local_files_only 加载）
```

也可独立对一台已租好的机器预取：

```bash
python -m tools.cloud_prefetch --host H --port P --ssh-key ~/.ssh/goodvision_vast \
  --remote-dir /workspace/goodvision-models \
  --model ZhuYuhan/VideoJev-4B@1c1a253bf90dfa66176124d25a48ffb791d517d5 \
  --model openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41
# R2_*/HF_TOKEN 从当前 shell 环境读取
```

机制通用：模型什么时候进了 R2 就从 R2 拉，没进就 HF 兜底——不绑定具体模型，也不改各 renter 的现有权重流程。

## 环境也打包进 R2（conda-pack，免编译起环境）

环境和权重是两个独立制品：权重仍由 `model_source` 填充 HF cache；环境由
`env_source` 恢复。环境读取失败会报错，不回退到 HF，也不会现场编译。

### 注册表与不可变对象

`tools/models.yaml` 固定每个环境的 `r2_key`、`manifest_key`、完整 SHA256、字节数、
`unpack_to`、`activate_rel` 与 `compat`。恢复必须具备全部字段；R2 边车 manifest 的
`format: conda-pack`、key、hash、size、prefix、activate 路径及兼容信息必须逐项一致。
远端 manifest 不能代替仓库中的可信 hash。现有 VideoJev pin 和权重对象保持不变。

新构建的对象使用内容地址：

```text
goodvision-models/
├── models/<org>/<name>/<revision>/      # 权重身份，不变
└── envs/<model>/<sha256>.tar.gz         # 新环境对象
    envs/<model>/<sha256>.json           # 同一制品的边车 manifest
```

`build_env` 使用 R2 原子 `PutObject(IfNoneMatch="*")`，不会覆盖旧对象，不使用
无条件 multipart 上传。上传上限保守设为 5 GiB 减 5 MiB；更大的包直接拒绝。
第二个写入失败或超时可能留下未被引用的 tarball，必须人工检查，不自动删除、
覆盖或追加无条件写入。输出的新 pin 需审查后更新注册表。
参考：[R2 条件写入](https://developers.cloudflare.com/r2/api/s3/extensions/#conditional-operations-in-putobject)、
[R2 大小限制](https://developers.cloudflare.com/r2/platform/limits/)。

### 一次性构建（仅维护者，在已授权的匹配 GPU 上）

默认 GPU 配方使用 Python 3.12 / torch 2.8.0+cu128 / CUDA 12.8。
要求 Linux x86_64、可用的 `nvidia-smi`、匹配的 GPU compute capability，以及驱动报告
的 CUDA 支持不低于 12.8。Ada 对应 8.9；Ampere 对应 8.0/8.6；Hopper 对应 9.0。
所有可见 GPU 都须匹配。同一家族检查不能替代实际 CUDA 内核验证。

先安装可信的 Miniforge；或者提供本地 installer 和独立核验过的 SHA256。
构建器不再下载并执行浮动的 `latest` installer。环境前缀和输出文件都必须不存在；
已有环境不会被删除。构建失败留下的前缀供维护者检查，重试前需主动移走。

```bash
# 仅在已授权的构建机上运行；读写 R2_* 从环境传入，不写进命令参数。
python -m tools.build_env --model videojev \
  --conda-home /opt/miniforge3 --out /tmp/videojev-new.tar.gz
# 只构建、不上传：加 --skip-upload
# 若尚未安装 Miniforge：加 --miniforge-installer /path/installer.sh \
#   --miniforge-sha256 <独立核验过的完整摘要>
```

构建器按 `--model` 选择默认 requirements，不会把 VideoJev 配方用于其他模型：

| 模型 | 默认配方 |
| --- | --- |
| `videojev` | `tools/videojev_probe/requirements.txt` |
| `clip` | `tools/clip/requirements.txt` |
| `siglip2` | `tools/siglip2/requirements.txt` |
| `d3-nano` | `tools/d3_nano/requirements.txt` |
| `qwen8b` | `tools/qwen8b/requirements.txt` |
| `qwen8b-vllm` | `tools/qwen8b_vllm/requirements.txt` |

`qwen8b-vllm` 使用独立的 CPU 构建路径：仅允许 Linux x86_64，按选定注册表条目
的 `env.compat` 验证 Python/torch/CUDA 和声明的扩展版本，不预装 HF 的 torch。
本分支保留的 cu128 条目声明 Python 3.12、torch 2.13.0、CUDA 12.8 和 vLLM 0.31.0；
这些声明不等于已验证的运行事实，也不能用 cu130/ada2 的结果证明。构建后继包前须
先审查新的配方与不可变身份，不能为使构建通过而改写历史条目。
`require_cuda: false` 会明确留下 GPU 验证待完成；不会把 CPU 构建当作 GPU 验收。
flash-attn 编译仍要求 GPU。构建与打包后检查 conda 文件完整性，并在全新临时
目录安全恢复，用包内隔离解释器导入声明的运行模块。源前缀的 pip RECORD
摘要差异只有在 conda 元数据同时绑定原摘要与安装后摘要时才接受；其余差异拒绝。
恢复后的 conda-unpack 会重写前缀，原 RECORD 摘要检查明确跳过，缺失文件和导入检查
继续执行；这不证明 ABI。缺少重定位来源的旧包仍需单独调查。pip 不升级 conda 管理的
setuptools；缺失文件不会通过 `--ignore-missing-files` 隐藏。

成功构建并打包后，会以独占创建方式写入 `<out>.build.json`。该记录绑定 tarball
的 SHA256/大小、模型 revision、目标 prefix、GPU 家族、flash-attn 开关、顶层
requirements 文件摘要、构建器源码摘要，以及包内解释器实测的 Python/torch/CUDA
版本、GPU compute capability 和所需扩展版本。构建或打包失败不会生成有效记录；
已有 tarball 或记录均不会被覆盖。

新构建使用版本 2 记录，绑定 torch 安装开关、必需导入模块及
`gpu_validation: deferred` 或 `measured`；原 GPU 配方仍接受已有版本 1 记录。
`--skip-pack` 只构建并写入 `<out>.prefix.json`；后续 `--pack-only` 必须使用同一
输出、模型、配方及该记录，先核对所有前缀文件内容、模式与链接指纹，再重新验证运行事实后才能打包。
记录保留原始构建器身份；打包过程中前缀变化会拒绝生成制品记录。它不会为任意旧前缀
补造来源记录。`--skip-pack`、`--pack-only`、`--upload-only` 互斥。

可以先构建，再在没有 GPU 的上传机上上传；须一起保留并移动 tarball 和同名记录：

```bash
python -m tools.build_env --model clip --conda-home /opt/miniforge3 \
  --out /tmp/clip-new.tar.gz --skip-upload
# 上传机需持有同一配方；只使用已授权的 R2 写凭据。
python -m tools.build_env --model clip \
  --out /tmp/clip-new.tar.gz --upload-only
```

`--requirements PATH` 可显式覆盖默认配方，后续 `--upload-only` 也必须提供内容
相同的配方。上传前核对构建记录、当前选定的模型/配方、tarball 摘要和归档必要文件；
不会执行 tarball，也不会用上传机的信息补造构建事实。缺少记录、错模型、改过的包、
损坏或不完整归档都会在创建 R2 客户端前拒绝。旧包没有这种记录时须重新构建，不能
靠手写记录获得验收。上传 manifest 会保留构建记录，供后续审查。

记录由受信任的构建流程产生，不是数字签名，也不是依赖锁文件或独立的 ABI 验收。
顶层 requirements 摘要不等于完整传递依赖锁定；主机/版本检查也不证明实际 CUDA
内核或模型推理可用。真实包的来源、恢复、依赖及 ABI 验收仍须单独完成。恢复端的
完整安全解包检查继续适用。条件写入失败或结果不确定时仍需检查远端对象，
`--upload-only` 不会覆盖既有对象或自动删除部分上传。

HF qwen8b 及三代 vLLM 历史/当前 key 均冻结，包括 `cu130-py312-vllm0310-ada2`。
对这些注册表身份，正常上传、`--pack-only` 上传和 `--upload-only` 均拒绝；先准备
经过审查的新身份。`--skip-upload` 可本地构建，输出不代表已发布或已验收。

构建/解包子进程不继承 R2/HF/cloud 凭据或 `PYTHONPATH`。路径通过 argv 传递；
`conda-pack` 不使用 `--ignore-missing-files` 或 `--force`。

### 每次使用：在远端恢复，再用包内解释器启动

远端需有 Python（带 `tarfile.data_filter` 的受维护版本）、`boto3`、`PyYAML`、
`tar`（仅用于上传小型 reader 文件）和 `nvidia-smi`。大包解压不调用系统 tar。
`cloud_env` 将 reader 文件上传到独立临时目录，R2 **只读** key 仅经 SSH stdin
传入，结束后清理 reader。返回的是远端 `{unpack_to}/bin/python`，不能在本机执行：
远端必须同时回报所选包的 key、SHA256、大小、前缀、兼容性及 manifest key，
并与控制机启动前捕获的身份完全一致；缺少身份或仅路径相同都不能作为成功回执。
只验证 stdout 最后一条非空回执；后续错误、错误路径或错误身份不能被较早的成功
输出掩盖。完成标记的旧格式保持不变。

```bash
# 在控制机运行；先通过安全的既有方式导出四个只读 R2_* 变量。
python -m tools.cloud_env --host H --port P --ssh-key KEY --model videojev
# 成功后会输出远端 /opt/env/videojev/bin/python。
# 另开远端 shell：
ssh -p P -i KEY root@H
# 以下命令只在远端 shell 中运行；不要 source activate，也不要用 PATH 上的 python：
PYTHONNOUSERSITE=1 /opt/env/videojev/bin/python -s -c 'import sys; assert sys.prefix == "/opt/env/videojev"'
```

恢复流程：校验注册表和主机 → 加前缀锁 → 核对 manifest → 私有目录下载 →
SHA256/size 校验 → 安全临时解包 → 放到固定前缀 → 用包内 `bin/python -I`
运行 `bin/conda-unpack` → 写完成标记 → 返回包内 `bin/python`。允许安全内部软/硬链接；
拒绝路径逃逸、链接父目录、设备/FIFO、重复路径和包内伪造的完成标记。
发布前缀使用原子 no-replace rename，不会创建等待确认所有权的空公开目录，也不会
替换其他进程刚创建的空目录。Linux 主机的 libc、内核及文件系统必须支持
[`renameat2(RENAME_NOREPLACE)`](https://man7.org/linux/man-pages/man2/rename.2.html)；
不支持时直接报错，不退回可能覆盖目标的普通 rename。macOS 的
[`RENAME_EXCL`](https://developer.apple.com/documentation/foundation/urlresourcekey/volumesupportsexclusiverenamingkey)
仅用于离线合成恢复测试，不代表 Linux/GPU 验收。

匹配的完成标记和必要文件允许本地复用；这不是对整个已使用环境的重新取证校验。
任何其他已有前缀（包括旧版本、损坏、旧格式标记或无标记目录）都被保留并拒绝覆盖。
需要更换时先检查并主动移走它。`--no-local` 同样要求空缺前缀，不会强制删除。
普通异常、Ctrl-C、SIGTERM/SIGHUP 清理本次下载和新前缀；SIGKILL/断电可能留下无
完成标记的目录，下一次会拒绝把它当作可用环境，需检查后移走。锁文件可以保留，
内核会在进程退出时释放锁。

### 验证范围

PR 作者报告原始 Ada/4090 包还原耗时 126 秒、可导入 torch/flash-attn 并识别 CUDA。
这是作者的历史报告，代码审查未独立复现。离线回归使用合成 tarball、假 R2/SSH
与主机检测 mock；不证明现存 4.85 GB 包的链接布局、ABI、真实 GPU 内核或速度。
作者还需用原始对象确认完整 manifest 与注册表一致，并在已授权的兼容主机上提供
更新后恢复路径的证据；若旧包不满足安全约束，应发布新身份，不能覆盖现有对象。

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
| 视觉集单 revision（VideoJev≈9 + CLIP≈1.7 + SigLIP2≈1.5 + d3-nano≈4.5） | ≈ 16.7 GB | ≈ $0.10（扣除 10GB 免费额度后约 6.7GB 计费） |
| 打包环境 `envs/`（VideoJev conda-pack，Ada/cu128） | ≈ 4.85 GB | ≈ $0.07/月；省掉每次租机现装 torch+flash-attn 的等待 |
| 视觉集 ×2 revision + 一个 Qwen FP8（≈35） | ≈ 59 GB | ≈ $0.89 |
| 宽松上限 | ≈ 100 GB | ≈ $1.50 |

- **操作成本可忽略**：上传一次全部分片约数百次 PUT（远低于 100 万免费）；每次租机拉取几十~几百次 GET，就算 100 次租机 ×500 GET = 5 万，也远低于 1000 万免费。
- **egress 免费是关键**：相比每次新租机从 HF 重拉（分钟级占用**付费 GPU** + HF 限速/偶发失败），或付 Vast 磁盘费，R2 的 S3 API + 免费 egress 让再水合又快又免费。

**对比 Vast 自带 storage：** Vast 的实例/卷磁盘按**宿主报价**计费（各 offer 不同，常见约 $0.05–0.20/GB-月），且**只在实例/卷存活期间**收费——实例一销毁就没了，于是下次又得重下。network volume 能持久但仍是宿主定价且绑定地区。真正的隐性成本不是磁盘费，而是**每次新租机从 HF 重下**：占用付费 GPU 的等待时间 + HF 不稳（正是本任务要解决的痛点）。R2 把权重钉在一处、egress 免费、S3 协议直连，按月几乎零成本就消除了这个重复下载问题。

## 待办与需要用户提供的信息

**已完成：** 账号（`sx2660@nyu.edu`，account `2c044451…`）+ bucket `goodvision-models` 确认；VideoJev + CLIP + SigLIP2 + d3-nano 已上传校验；只读 token `vast-R2-goodvision-models-readonly` 已建并验证（能读、写被拒），4 个 `R2_*` 写入本地 `.env`（未提交）；runner 集成（`tools/model_source.py` + `videojev_probe download`）已落地并对真实 R2 冒烟通过。

**待办：**

1. **其余模型上传**（用读写 key，维护者执行）：
   - ~~d3-nano~~ **已上传 ✅**（`vllm-sr/d3-nano` @ `462efc36…`，4.46 GB）。
   - ~~SigLIP2~~ **已上传 ✅**（`google/siglip2-base-patch16-224` @ `75de2d55…`，1.54 GB；revision 钉在 `tools/model_source.py` `_KNOWN`，服务接入时再移到运行时模块）。
   - **Qwen3.5-35B FP8**（`Qwen/Qwen3.5-35B-A3B-FP8` @ `9d1823d2…`，~35GB）——确认要不要进（建议要；BF16 暂不）。
   - Qwen3-VL-8B 是纯 API baseline，无本地权重，不进 R2。
2. **只读凭据分发**：把那 4 个 `R2_*` 放进团队密码管理器，供其他同事 runner 用（**不要进 repo/聊天**）。
3. **持久卷**：等 Qwen FP8 上了，给它在 Vast 挂持久卷以免重租重下。
4. **Vast cloud connection（可选）**：若想用 `vastai cloud copy` 而非 boto3/rclone，在网页控制台把 R2 注册成 S3 连接。
