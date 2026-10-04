# ruff: noqa: RUF001
"""Self-contained HTML plus portable JSON; never promote mock results to real."""

import html
import json
import os
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def render(directory, document):
    directory = Path(directory)
    write_json(directory / "summary.json", document)
    e = lambda s: html.escape(str(s))  # noqa: E731
    simulated = document["plan"]["simulate"]
    title = "模拟调度测试（未调用真实 VLM）" if simulated else "纯 VLM 多路容量测试"
    if document["plan"].get("compare_concurrency") and not simulated:
        title = "纯 VLM 单路并发 2/4 对照"
    if document["plan"].get("compare_arrival") and not simulated:
        title = "纯 VLM 同步/错峰输入对照"
    if document["plan"].get("compare_routing") and not simulated:
        title = "纯 VLM 吞吐优先/低延迟优先路由对照"
    rows = []
    details = []
    for job in document["jobs"]:
        stage = job["result"]
        coverage = [r["on_time_coverage"] or 0 for r in stage["per_stream"]]
        p95 = [
            r["source_to_publish_seconds"]["p95"]
            for r in stage["per_stream"]
            if r["source_to_publish_seconds"]["p95"] is not None
        ]
        latency = f"{min(p95):.2f}–{max(p95):.2f} s" if p95 else "无有效结果"
        passed = "符合实验门槛" if job["gate"]["passed"] else "不符合实验门槛"
        if job["phase"] == "probe":
            passed = (
                "接口可用（不代表性能达标）"
                if job["gate"]["passed"]
                else "接口探测失败"
            )
        rows.append(
            f"<tr><td>{e(job['id'])}</td><td>{job['streams']}</td>"
            f"<td>{min(coverage):.1%}–{max(coverage):.1%}</td>"
            f"<td>{latency}</td><td>{e(passed)}</td></tr>"
        )
        camera_rows = []
        for camera, r in enumerate(stage["per_stream"], 1):
            camera_rows.append(
                f"<tr><td>{camera}</td><td>{r['expected_observations']}</td>"
                f"<td>{r['complete_valid_results']}</td>"
                f"<td>{r['on_time_valid_results']}</td>"
                f"<td>{e(r['uncertain_judgment_fraction'])}</td>"
                f"<td>{e(json.dumps(r['errors']))}</td>"
                f"<td>{e(json.dumps(r.get('http_status_counts', {})))}</td></tr>"
                f"<tr><td colspan='7'>返回校验诊断："
                f"{e(json.dumps(r.get('validation_reason_counts', {})))}</td></tr>"
                f"<tr><td colspan='7'>回复结构（不含原文）："
                f"{e(json.dumps(r.get('response_diagnostic_counts', {})))}</td></tr>"
                f"<tr><td colspan='7'>有效结果的外层格式："
                f"{e(json.dumps(r.get('accepted_wrapper_counts', {})))}</td></tr>"
                f"<tr><td colspan='7'>请求分段统计（时长单位秒，bytes/tokens 按字段名；"
                f"含失败请求的已观察部分）：<pre>"
                f"{e(json.dumps(r.get('request_phase_metrics', {}), indent=2))}"
                "</pre></td></tr>"
            )
        details.append(
            f"<details><summary>{e(job['id'])}：逐路结果</summary>"
            "<table><tr><th>路</th><th>计划</th><th>完整结果</th>"
            "<th>及时结果</th><th>不确定比例</th><th>错误</th>"
            f"<th>HTTP 状态</th></tr>{''.join(camera_rows)}</table>"
            f"<p>{e('; '.join(job['gate']['reasons']))}</p></details>"
        )
    p = document["plan"]
    arrival_note = ""
    if p.get("compare_routing"):
        arrival_note = (
            "<p class='notice'>同一基础模型，"
            "按吞吐优先→低延迟优先→低延迟优先→吞吐优先对照。"
            "保留相同视频、帧、提示、schema、输入频率和期限；不使用 Nitro 等路由后缀。"
            "仅修改 provider.sort，各轮实际请求参数记录于 summary.json；"
            "上游路由和网络仍可能变化，短测不代表持续容量或事件准确率。</p>"
        )
    if p.get("compare_arrival") or p.get("arrival_mode") == "staggered":
        arrival_note += (
            "<p class='notice'>错峰方式：将各路的源时钟均匀错开，覆盖一个观察周期。"
            "各路保留相同视频帧、源时间戳、采样频率和完整源时长，"
            "不丢尾帧，也不缩小计划窗口分母。跨路总送帧区间略长于单路时长；"
            "逐路起始偏移与总区间记录于 summary.json。"
            "每路单独判定，不代表生产调度的所有问题已经修复。"
        )
        if p.get("compare_arrival"):
            arrival_note += (
                "对照顺序为同步→错峰→错峰→同步，各两轮，检验同步突发的影响。"
            )
        arrival_note += "</p>"
    comparison = ""
    if p.get("compare_concurrency"):
        arms = []
        for arm in document.get("concurrency_comparison", []):
            errors = arm["errors"]
            coverage = arm["on_time_coverage"]
            percent = f"{coverage:.1%}" if coverage is not None else "尚无结果"
            status = (
                "未完成，不能判定"
                if not arm["complete"]
                else "两轮均符合门槛"
                if arm["all_trials_passed"]
                else "至少一轮不符合门槛"
            )
            queue_drops = sum(
                errors.get(code, 0)
                for code in ("admission_superseded", "admission_timeout")
            )
            arms.append(
                f"<tr><td>{arm['per_stream_limit']}</td>"
                f"<td>{arm['completed_trials']}/{arm['required_trials']}</td>"
                f"<td>{arm['expected_observations']}</td>"
                f"<td>{arm['complete_valid_results']}</td>"
                f"<td>{arm['on_time_valid_results']} / {percent}</td>"
                f"<td>{errors.get('invalid_response', 0)}</td>"
                f"<td>{errors.get('timeout', 0)}</td><td>{queue_drops}</td>"
                f"<td>{e(status)}</td></tr>"
            )
        comparison = (
            "<h2>单路并发对照</h2><p>顺序为 2 → 4 → 4 → 2，各两轮。"
            "并发数是同一路同时在途的模型请求上限，不是视频路数或 batch 大小。"
            "两组使用相同输入、模型、采样频率、输出格式与时效门槛；"
            "外部 API 的具体上游路由和网络状态仍可能变化。"
            "本次只测一路，不能据此声称十二路通过。</p>"
            "<table><tr><th>每路并发上限</th><th>完成轮次</th><th>计划窗口</th>"
            "<th>完整有效结果</th><th>及时结果 / 比例</th><th>返回校验失败</th>"
            "<th>请求超时</th><th>排队替换或超时</th><th>判断</th></tr>"
            + "".join(arms)
            + "</table><p>比例按两轮的计划窗口数合并；不能用合并比例掩盖某轮失败。"
            "未排空或送帧无效的阶段仍显示诊断数据，但不会计为通过。"
            "各轮延迟分别列于下表；不平均 P95。</p>"
        )
    body = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>{e(title)}</title><style>
body{{font:16px/1.6 system-ui;max-width:1150px;margin:36px auto;
padding:0 24px;color:#183048}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}}
td,th{{text-align:left;padding:9px;border:1px solid #d4dfe7}}
th{{background:#edf3f8}}.notice{{background:#fff5d7;padding:16px;border-radius:8px}}
details{{margin:18px 0}}code,pre{{white-space:pre-wrap;word-break:break-word}}
</style><h1>{e(title)}</h1>
<p class="notice">状态：{e(document["state"])} ·
{e(document.get("conclusion", "测试尚未完成"))}<br>
本报告验证请求容量和时效，不代表事件识别真实准确率。没有 SAM 或 GPU 推理参与事件识别；
模型通过外部 API 调用。视频预解码后按实际时间送帧，排除摄像头上传和浏览器显示。
同一视频复制为十二路，只是负载测试，不等于十二个不同场景的识别验收。</p>
<p>模型：{e(document.get("model", "尚未载入"))}；提供商：{e(p["provider"])}；
上游限定：{e(p.get("provider_only") or "自动选择")}；
每路 {p["observation_fps"]} 个观察窗口/秒。
一个请求含同一路最近 2–4 帧，保留动作上下文。</p>
<p>输出约束：{e(p.get("event_json_mode", "prompt"))}。
schema 表示请求四个事件字段的 JSON Schema 约束；仍进行本地严格校验。
仅允许剥离完整的外层 JSON 代码块，不补字段、不修补截断文本，也不把 null 改成 false。</p>
<p>预先固定的实验门槛：每路 {p["deadline"]} 秒内完整结果比例 ≥ {p["min_coverage"]:.0%}，
请求错误比例 ≤ {p["max_error_fraction"]:.0%}，
不确定判断比例 ≤ {p["max_uncertain_fraction"]:.0%}。
这是可配置的工程筛选门槛，不是已由会议确认的验收标准。</p>
{comparison}
{arrival_note}
<table><tr><th>阶段</th><th>路数</th><th>逐路及时完整结果比例</th>
<th>逐路 P95（最新证据入分支→发布）</th><th>判断</th></tr>{"".join(rows)}</table>
<p>未完成或中断的阶段不会计为通过。计划观察数包含被跳过、超时和缺失的窗口；
probe 是一次隔离请求，仅验证接口能否返回完整结果；它不参与性能验收。
字段完整但判断为 null 会另计“不确定”。上述延迟不等于事件开始到首次识别的时间。</p>
<p>请求分段中部分计时互相包含，不能直接相加。
upload_to_first_non_whitespace_seconds 包含上传完成后的网络及上游等待、推理等，
不等于纯模型计算时间。缺失指标表示未观察到该阶段，不表示耗时为零。</p>
{"".join(details)}<h2>失败或跳过原因</h2><pre>{e(document.get("reason", "无"))}</pre>
<p><a href="summary.json">完整汇总 JSON</a> ·
<a href="manifest.json">环境与输入清单</a> ·
<a href="telemetry.jsonl">CPU/内存遥测</a></p></html>"""
    target = directory / "report.html"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, target)
