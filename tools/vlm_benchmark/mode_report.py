# ruff: noqa: RUF001
"""Three-arm report with independent denominators and observable timings."""

import html
import json
from pathlib import Path

from tools.vlm_benchmark.report import write_json


def recording_links(output, job):
    """Only link artifacts that were actually written, including partial WebM."""
    recording = job.get("recording")
    if not recording:
        return ""
    folder = Path(output) / job["id"] / "recording"
    available = [
        (folder / "viewport.mp4", "MP4"),
        (folder / "recording.json", "录制完整性检查"),
        (folder / "final.png", "最终画面"),
        *[(p, "原始录像 / 可能不完整") for p in sorted(folder.glob("*.webm"))],
    ]
    links = []
    for p, label in available:
        if p.is_file():
            href = html.escape(p.relative_to(output).as_posix())
            links.append(f"<a href='{href}'>{label}</a>")
    complete = recording.get("complete") and (folder / "viewport.mp4").is_file()
    status = "完整" if complete else "不完整/失败"
    playback = recording.get("playback_quality")
    playback_note = ""
    if playback:
        rates = [row["fps"] for row in playback["per_stream"] if row["fps"] is not None]
        measured = f"{min(rates):.2f}–{max(rates):.2f} FPS" if rates else "无有效测量"
        playback_note = (
            f" · 逐路实际播放：{measured}；目标 15 FPS，"
            "帧率仅作记录，不作为中止条件"
        )
        if not playback.get("measurement_complete", True):
            playback_note += "；部分播放测量数据缺失"
    return (
        f"<p>浏览器录屏：{status}{playback_note} · "
        + (" · ".join(links) or "尚无可用文件")
        + "</p>"
    )


def render_modes(output, document):
    output = Path(output)
    write_json(output / "summary.json", document)
    sweep = document["settings"].get("grounding_concurrency_sweep")
    if sweep:
        from tools.vlm_benchmark.grounding_sweep_report import render

        render(output, document)
    if document["settings"].get("parallel_concurrency_sweep"):
        from tools.vlm_benchmark.parallel_sweep_report import render

        render(output, document)
    e = lambda value: html.escape(str(value))  # noqa: E731
    fmt = lambda value: "不可用" if value is None else f"{value:.3f}"  # noqa: E731
    rows, details, recordings, accounting_rows = [], [], [], []
    for job in document["jobs"]:
        stage = job["result"]
        valid = (
            stage["all_drained"]
            and stage["input_timing_valid"]
            and not stage["result_records_missing"]
            and stage.get("result_retention", {}).get("valid") is not False
            and not (stage.get("shared_admission") or {}).get("active", 0)
            and not (stage.get("shared_admission") or {}).get("waiting", 0)
            and job.get("capacity", {}).get("valid", True)
            and job.get("route_audit", {}).get("valid", True)
        )
        for lane, data in stage["lanes"].items():
            accounting = data.get("delivery_accounting", {})
            accounting_rows.append(
                f"<tr><td>{e(job['id'])}</td><td>{e(lane)}</td>"
                + "".join(
                    f"<td>{e(accounting.get(key, '未记录'))}</td>"
                    for key in (
                        "http_attempted", "http_attempted_complete",
                        "http_attempted_without_complete", "http_not_attempted_records",
                        "http_attempt_unknown_records", "planned_without_record",
                        "historical_complete_results", "historical_failed_reports",
                        "nonhistorical_complete_results",
                    )
                )
                + f"<td>{e(stage.get('result_retention', {}).get('valid', '未记录'))}"
                "</td></tr>"
            )
            n = data["expected_observations"]
            cdf = " / ".join(
                f"{data['within_seconds'][str(t)]['count']}/{n}" for t in (1, 2, 3, 5)
            )
            rows.append(
                f"<tr><td>{e(job['phase'])}</td><td>{e(job['mode'])}</td>"
                f"<td>{stage['streams']}</td><td>{stage['repeat']}</td><td>{e(lane)}</td>"
                f"<td>{data['complete_valid_results']}/{n}</td>"
                f"<td>{data['without_complete_result']}</td>"
                f"<td>{fmt(data['source_to_publish_seconds']['p50'])}</td>"
                f"<td>{fmt(data['source_to_publish_seconds']['p95'])}</td>"
                f"<td>{cdf}</td><td>{'有效' if valid else '无效，保留诊断'}</td></tr>"
            )
        cameras = []
        for index, camera in enumerate(stage["per_stream"], 1):
            for lane, data in camera["lanes"].items():
                cameras.append(
                    f"<tr><td>{index}</td><td>{e(lane)}</td>"
                    f"<td>{data['complete_valid_results']}/{data['expected_observations']}</td>"
                    f"<td>{fmt(data['source_to_publish_seconds']['p50'])}</td>"
                    f"<td>{fmt(data['source_to_publish_seconds']['p95'])}</td>"
                    f"<td>{e(data['errors'])}</td>"
                    f"<td>{data['localization_complete_results']}</td>"
                    f"<td>{data['all_negative_results']}</td></tr>"
                )
        phases = {
            lane: {
                **data["request_phase_metrics"],
                "source_to_completion_seconds": data.get(
                    "source_to_completion_seconds"
                ),
                "completion_to_publish_seconds": data.get(
                    "completion_to_publish_seconds"
                ),
            } for lane, data in stage["lanes"].items()
        }
        recording_link = recording_links(output, job)
        if recording_link:
            recordings.append(
                f"<div><strong>{e(job['id'])}</strong>{recording_link}</div>"
            )
        details.append(
            f"<details><summary>{e(job['id'])}：逐路、配对和耗时分段</summary>"
            f"{recording_link}"
            "<table><tr><th>视频路</th><th>分支</th><th>完整/计划</th><th>P50 秒</th>"
            "<th>P95 秒</th><th>错误</th><th>定位完整</th><th>全部阴性</th></tr>"
            + "".join(cameras)
            + "</table>"
            f"<p>两分支同窗口配对：{e(stage['paired_windows'])}</p>"
            f"<p>逐路结果留存核对：{e(stage.get('result_retention', '未记录'))}</p>"
            f"<p>模型与服务商核对：{e(job.get('route_audit', '未启用'))}</p>"
            f"<p>计划请求率：{stage['nominal_requests_per_second']:.2f}/秒；"
            f"发起上限低于负载：{e(stage['rate_limit_below_offered_load'])}</p>"
            f"<pre>{e(json.dumps(phases, ensure_ascii=False, indent=2))}</pre>"
            "<p>下方为整个推理服务的阶段平均耗时，不能归到某条分支，也不是 P95。</p>"
            f"<pre>{e(json.dumps(job['telemetry'], indent=2))}</pre>"
            f"<a href='{e(job['raw_reports'])}'>逐请求记录</a> · "
            f"<a href='{e(job['telemetry_file'])}'>服务与 GPU 采样</a></details>"
        )
    for job in document.get("interrupted_jobs", []):
        details.append(
            f"<p>{e(job['id'])} 未完成：{e(job['error_type'])}。"
            "没有完整统计，不能按成功轮次计入。</p>"
        )
        link = recording_links(output, job)
        if link:
            recordings.append(
                f"<div><strong>{e(job['id'])}（未完成）</strong>{link}</div>"
            )
    scope = (
        "模拟调度，未调用模型"
        if not document["settings"]["real"]
        else "自部署端点，硬件与模型精度见 manifest；GPU 采样只对应运行脚本的主机"
        if document["settings"]["backend"] == "local"
        else "阿里云百炼直连 API；本机 GPU 不执行 VLM 推理"
        if document["settings"]["backend"] == "dashscope"
        else "外部 OpenRouter API；本机 GPU 不执行 VLM 推理"
    )
    monitor_on = bool(
        document["settings"].get("monitor") or document["settings"].get("record")
    )
    format_note = (
        "legacy 全阴性可返回简短结果；需要定位时输出完整结构。"
        if document["settings"].get("grounding_format", "legacy") == "legacy"
        else "lean-v1 使用精简字段，仍逐类返回六类判断与人物引用。"
    )
    sweep_link = (
        '<p><a href="grounding-concurrency.html">Grounding 并发对照汇总</a></p>'
        if sweep else ""
    )
    if document["settings"].get("parallel_concurrency_sweep"):
        sweep_link += (
            '<p><a href="parallel-concurrency.html">'
            'Parallel 配对与并发对照汇总</a></p>'
        )
    page = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>VLM 三模式耗时对照</title><style>
body{{font:16px/1.6 system-ui;margin:32px auto;padding:0 24px;
max-width:1500px;color:#183048}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border:1px solid #ccd8e2;padding:8px;text-align:left}}
th{{background:#edf3f8}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
details{{margin:20px 0}}.note{{padding:16px;background:#fff5dc}}
</style><h1>VLM：events / compact / parallel</h1>
<p class="note">状态：{e(document["state"])}；{e(scope)}。
{e(document.get("reason", ""))}</p>
<p>调用渠道：{e(document["settings"]["backend"])}；
请求模型：{e(document["settings"].get("model", "未记录"))}。</p>
<p>固定服务商：{e(document["settings"].get("provider_only") or "未固定")}；
优先服务商：{e(document["settings"].get("provider_prefer") or "未指定")}；
允许其他供应商接替：{e(not bool(document["settings"].get("provider_only")))}；
监看布局：{e(document["settings"].get("monitor_layout", "evidence"))}；
视频预览目标 15 FPS，与模型请求频率分开；录屏编码设置为 25 FPS，
实际播放帧率见录制诊断。</p>
{sweep_link}
<p>正在执行或中断的阶段：{e(document.get("active_stage", "无"))}；
未满足筛选条件而跳过的持续测试：{e(document.get("skipped_soaks", []))}。</p>
<p>实时监看：{e(monitor_on)}；
浏览器页面录屏：{e(bool(document["settings"].get("record")))}。
启用预览与录屏会增加测试机负载，输入时序仍单独检查。
录屏为测试监看页面，不代表完整产品界面的端到端性能。</p>
<p>parallel 已包含 events 和 compact，不另加 events 请求。
六类事件：{e(", ".join(document["event_labels"]))}。
不使用 SAM。compact 是 grounding 单独运行；parallel 同时运行 events 与 grounding。
定位格式：{e(document["settings"].get("grounding_format", "legacy"))}。
{format_note}
不能将全阴性当作定位失败。</p>
{"<h2>本次录屏</h2>" + "".join(recordings) if recordings else ""}
<p>延迟从最新证据进入 VLM 分支到结果发布，排除解码、摄像头上传和显示。
P50/P95 使用该分支全部完整结果，不平均逐路分位数。
1/2/3/5 秒仅是分布参考，未设业务达标结论。
失败、跳过、未返回均保留在计划窗口分母中；预检、预热和正式测量分开列示。</p>
<p>每视频路的 admission 并发额度由两分支共享；本地线程上限则按分支设置。
不是把多路合成一个图片 batch。相同视频复制测试负载，不验证独立场景识别准确率。
固定量化、图片、thinking、缓存设置才能比较不同轮次。</p>
<table><tr><th>阶段</th><th>模式</th><th>路数</th><th>重复</th><th>分支</th>
<th>完整/计划</th><th>无完整结果</th><th>P50 秒</th><th>P95 秒</th>
<th>≤1/2/3/5 秒，各自/计划</th><th>输入/排空</th></tr>
{"".join(rows)}</table>
<h2>调用与结果留存</h2>
<p>所有完整返回（包括历史结果）计入返回率。历史结果只保留供查询和评分，不覆盖更新的实时状态；
非历史完整结果也不等于浏览器实际显示过的次数。
HTTP 已尝试指客户端开始 send，不保证服务商已收到。
历史失败是否发起调用仍以 HTTP 标记为准，不能仅凭迟到或失败归类。
没有调用标记的旧记录显示为未知；
计划中无记录的窗口单列，不能仅用“计划数−保存的 HTTP 数”断言全部未发出。
逐路按已提交、已结束、已发布和已留存进行核对，核对失败停止后续轮次。
模拟测试不发 HTTP，请结合上方实测/模拟标识解读。</p>
<div style="overflow-x:auto"><table><tr><th>阶段</th><th>分支</th><th>HTTP 已尝试</th>
<th>其中完整返回</th><th>其中无完整结果</th><th>记录确认未尝试 HTTP</th>
<th>HTTP 状态未知</th>
<th>计划中无记录</th><th>历史完整结果</th><th>历史失败</th><th>非历史完整结果</th><th>留存核对</th></tr>
{"".join(accounting_rows)}</table></div>
<p>分段计时部分互相包含，不能直接相加。model_seconds 包含请求准备、HTTP 和解析，
不是纯 GPU 时间。first_non_whitespace 不是首个模型 token。
没有暴露的服务端指标显示不可用。
定位完整仅指输出结构和引用可用，不代表框正确；误报/漏报需另做标注对照。</p>
{"".join(details)}<p><a href="manifest.json">环境与固定配置</a> ·
<a href="summary.json">完整 JSON</a></p></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
