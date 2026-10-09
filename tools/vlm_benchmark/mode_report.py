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
    return (
        f"<p>浏览器录屏：{status} · " + (" · ".join(links) or "尚无可用文件") + "</p>"
    )


def render_modes(output, document):
    output = Path(output)
    write_json(output / "summary.json", document)
    e = lambda value: html.escape(str(value))  # noqa: E731
    fmt = lambda value: "不可用" if value is None else f"{value:.3f}"  # noqa: E731
    rows, details, recordings = [], [], []
    for job in document["jobs"]:
        stage = job["result"]
        valid = (
            stage["all_drained"]
            and stage["input_timing_valid"]
            and not stage["result_records_missing"]
        )
        for lane, data in stage["lanes"].items():
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
            lane: data["request_phase_metrics"] for lane, data in stage["lanes"].items()
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
        else "外部 OpenRouter API；本机 GPU 不执行 VLM 推理"
    )
    monitor_on = bool(
        document["settings"].get("monitor") or document["settings"].get("record")
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
<p>正在执行或中断的阶段：{e(document.get("active_stage", "无"))}；
未满足筛选条件而跳过的持续测试：{e(document.get("skipped_soaks", []))}。</p>
<p>实时监看：{e(monitor_on)}；
浏览器页面录屏：{e(bool(document["settings"].get("record")))}。
启用预览与录屏会增加测试机负载，输入时序仍单独检查。
录屏为测试监看页面，不代表完整产品界面的端到端性能。</p>
<p>parallel 已包含 events 和 compact，不另加 events 请求。
六类事件：{e(", ".join(document["event_labels"]))}。
不使用 SAM。compact 全阴性可返回简短结果；需要定位时输出完整结构。
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
<p>分段计时部分互相包含，不能直接相加。model_seconds 包含请求准备、HTTP 和解析，
不是纯 GPU 时间。first_non_whitespace 不是首个模型 token。
没有暴露的服务端指标显示不可用。
定位完整仅指输出结构和引用可用，不代表框正确；误报/漏报需另做标注对照。</p>
{"".join(details)}<p><a href="manifest.json">环境与固定配置</a> ·
<a href="summary.json">完整 JSON</a></p></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
