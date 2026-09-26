"""Scan explicit display delays; stop each model at its first repeated pass."""
# ruff: noqa: RUF001

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .__main__ import MODELS, ROOT
from .__main__ import main as run_single
from .service import display_delay


def parse_delays(value):
    try:
        values = [display_delay(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or values != sorted(set(values)):
        raise argparse.ArgumentTypeError("Delays must be unique and ascending")
    return values


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def repeated_pass(summary, model, delay, repeats):
    trials = summary.get("trials", [])
    return (
        len(trials) == repeats
        and {t.get("repeat") for t in trials} == set(range(1, repeats + 1))
        and all(
            t.get("valid") is True
            and t.get("model") == model
            and t.get("configured_delay_seconds") == delay
            and t.get("coverage", 0) >= 0.7
            and t.get("coverage_pass") is True
            for t in trials
        )
    )


def write_batch_report(batch):
    plan = json.loads((batch / "plan.json").read_text())
    records = {}
    for model in plan["models"]:
        path = batch / model / "sweep.json"
        if path.exists():
            records[model] = json.loads(path.read_text())
    rows, details = [], []
    labels = {
        "pending": "待运行",
        "running": "运行中",
        "passed": "两轮达标",
        "not_reached": "所测档位未达标",
        "failed": "失败 / 中断",
    }
    for model in plan["models"]:
        record = records.get(model, {})
        minimum = record.get("lowest_tested_passing_delay_seconds")
        steps = record.get("steps", [])
        tested = ", ".join(f"{step['delay']:g}" for step in steps) or "—"
        result = "—" if minimum is None else f"{minimum:g} 秒"
        rows.append(
            f"<tr><td>{model}</td><td>{tested}</td><td>{result}</td>"
            f"<td>{labels[record.get('status', 'pending')]}</td></tr>"
        )
        for step in steps:
            trials = step.get("summary", {}).get("trials", [])
            coverage = (
                " / ".join(
                    f"{t['coverage']:.1%}"
                    if t.get("coverage") is not None
                    else "无结果"
                    for t in trials
                )
                or "无结果"
            )
            model_times = (
                " / ".join(
                    f"{t['model_inference_seconds']['p50'] * 1000:.2f}"
                    if t.get("model_inference_seconds", {}).get("p50") is not None
                    else "—"
                    for t in trials
                )
                or "—"
            )
            validity = (
                "有效"
                if len(trials) == plan["repeats"]
                and all(t.get("valid") for t in trials)
                else "未完成 / 无效"
            )
            report = html.escape(step["report"], quote=True)
            link = (
                f'<a href="{report}">单档报告</a>'
                if (batch / step["report"]).exists()
                else "报告未生成"
            )
            details.append(
                f"<tr><td>{model}</td><td>{step['delay']:g}</td>"
                f"<td>{coverage}</td><td>{model_times}</td>"
                f"<td>{validity}</td><td>{link}</td></tr>"
            )
    complete = sum(r["status"] in {"passed", "not_reached"} for r in records.values())
    save(batch / "summary.json", {"plan": plan, "models": records})
    (batch / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>测试 3 · 延迟扫描汇总</title><style>"
        "body{font:15px/1.7 system-ui;color:#193047;background:#f1f5f8;"
        "max-width:1200px;margin:32px auto;padding:24px}"
        "section{background:white;padding:24px;border-radius:12px;margin:20px 0;"
        "overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}"
        "th,td{text-align:left;padding:12px;border-bottom:1px solid #dce4eb}"
        "th{background:#eaf0f5}a{color:#146b63}p{max-width:1000px}</style>"
        "<h1>测试 3 · 达到 70% 描边率的延迟扫描</h1>"
        f"<p>完成型号：{complete} / {len(plan['models'])}。"
        f"扫描档位（秒）：{html.escape(str(plan['delays']))}；"
        f"每档 {plan['repeats']} 轮，所有轮次有效且各自 ≥70% 才达标。</p>"
        "<p>按延迟从小到大测试，首个重复达标档位之后停止该型号。"
        "记录的是<strong>所测档位中的最低达标缓冲</strong>，不是精确的全局最短延迟；"
        "未测档位不作推断。两轮不代表长期可靠性，扫描不保证描边率单调上升。</p>"
        "<p>仅显式改变 main 的显示缓冲；模型、采样、队列及其他配置保持不变。"
        "GPU 分割在云端，Harness、人脸 CPU 与浏览器仍在 Mac。"
        "本批次独立保存，不混入早期 0 秒试跑或 SAM 历史数据。</p>"
        "<section><h2>各型号结果</h2><table><thead><tr><th>型号</th>"
        "<th>已尝试缓冲 / 秒</th><th>最低重复达标缓冲</th><th>状态</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
        "<section><h2>档位明细</h2><table><thead><tr><th>型号</th>"
        "<th>缓冲 / 秒</th><th>各轮描边率</th><th>各轮模型 P50 / ms</th>"
        "<th>轮次有效性</th><th>详细结果</th></tr></thead><tbody>"
        + "".join(details)
        + "</tbody></table></section>"
        "<p>描边率不是分割准确率，漏检可能抬高覆盖率。模型 P50 仅为 worker "
        "调用耗时，不等于达标端到端延迟；浏览器呈现耗时包含无描边帧，见单档报告。"
        "本轮不重跑 SAM，不计算相对 SAM 的直接加速倍数。</p></html>"
    )


def run_sweep(args, runner=run_single):
    batch = args.batch_dir.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    plan = {
        "models": args.expected_models,
        "delays": args.delays,
        "repeats": args.repeats,
    }
    plan_path = batch / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Batch plan differs; use a new batch directory")
    save(plan_path, plan)
    directory = batch / args.model
    directory.mkdir(exist_ok=False)
    record = {
        "model": args.model,
        "status": "running",
        "steps": [],
        "lowest_tested_passing_delay_seconds": None,
    }

    def persist():
        save(directory / "sweep.json", record)
        write_batch_report(batch)

    persist()
    try:
        for delay in args.delays:
            output = directory / f"delay-{delay:g}s"
            step = {
                "delay": delay,
                "report": str((output / "report.html").relative_to(batch)),
            }
            record["steps"].append(step)
            persist()
            command = [
                "--model",
                args.model,
                "--endpoint",
                args.endpoint,
                "--repeats",
                str(args.repeats),
                "--display-delay",
                str(delay),
                "--output",
                str(output),
            ]
            if args.allow_face_download:
                command.append("--allow-face-download")
            print(f"{args.model}: testing {delay:g} s buffer", flush=True)
            try:
                runner(command)
            finally:
                path = output / "summary.json"
                if path.exists():
                    step["summary"] = json.loads(path.read_text())
            summary = step.get("summary", {})
            trials = summary.get("trials", [])
            if len(trials) != args.repeats or not all(t.get("valid") for t in trials):
                raise RuntimeError("Invalid/incomplete trial; stopping delay scan")
            if repeated_pass(summary, args.model, delay, args.repeats):
                record["status"] = "passed"
                record["lowest_tested_passing_delay_seconds"] = delay
                persist()
                print(
                    f"{args.model}: repeated >=70% at {delay:g} s; stopping",
                    flush=True,
                )
                break
            persist()
        else:
            record["status"] = "not_reached"
            persist()
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        record["status"] = "failed"
        record["error_type"] = type(error).__name__
        persist()
        raise
    print(f"Delay-scan report: {batch / 'report.html'}", flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--delays", type=parse_delays, default=parse_delays("0.5,1,1.5,2")
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8092")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--allow-face-download", action="store_true")
    parser.add_argument("--expected-models", type=lambda s: s.split(","))
    parser.add_argument("--batch-dir", type=Path)
    args = parser.parse_args(argv)
    if not 2 <= args.repeats <= 10:
        parser.error("Delay scan requires 2..10 repetitions per setting")
    args.expected_models = args.expected_models or [args.model]
    if (
        args.model not in args.expected_models
        or len(set(args.expected_models)) != len(args.expected_models)
        or any(model not in MODELS for model in args.expected_models)
    ):
        parser.error("Expected models must be unique valid models including this model")
    args.batch_dir = args.batch_dir or ROOT / "outputs/seg-delay-sweep" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run_sweep(args)


if __name__ == "__main__":
    main()
