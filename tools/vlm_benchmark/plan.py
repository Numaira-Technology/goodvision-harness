"""Frozen load plan; engineering gates are not event-accuracy acceptance."""

import math


def validate(plan):
    def number(name, low, high, integer=False):
        value = plan[name]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (integer and type(value) is not int)
            or not low <= value <= high
        ):
            raise ValueError(f"invalid plan field: {name}")

    for name, low, high, integer in [
        ("duration", 3, 300, False),
        ("repeats", 1, 3, True),
        ("soak_seconds", 0, 1800, True),
        ("global_limit", 1, 128, True),
        ("per_stream_limit", 1, 4, True),
        ("request_rate", 0.1, 200, False),
        ("queue_timeout", 0.01, 3, False),
        ("observation_fps", 0.25, 3, False),
        ("deadline", 0.1, 30, False),
        ("min_coverage", 0.1, 1, False),
        ("max_error_fraction", 0, 1, False),
        ("max_uncertain_fraction", 0, 1, False),
    ]:
        number(name, low, high, integer)
    if plan["per_stream_limit"] > plan["global_limit"]:
        raise ValueError("per-stream limit exceeds global capacity")
    if (
        not isinstance(plan["streams"], list)
        or not plan["streams"]
        or any(type(n) is not int or not 1 <= n <= 12 for n in plan["streams"])
        or sorted(set(plan["streams"])) != plan["streams"]
    ):
        raise ValueError("streams must be an increasing unique subset of 1..12")
    if plan["provider"] not in {"openrouter", "dashscope"}:
        raise ValueError("this runner uses an external VLM API")
    if type(plan["compare"]) is not bool or type(plan["simulate"]) is not bool:
        raise ValueError("invalid plan flags")
    if type(plan.get("compare_concurrency", False)) is not bool:
        raise ValueError("invalid concurrency comparison flag")
    if type(plan.get("compare_arrival", False)) is not bool:
        raise ValueError("invalid arrival comparison flag")
    if type(plan.get("compare_routing", False)) is not bool:
        raise ValueError("invalid routing comparison flag")
    if plan.get("provider_sort", "") not in {"", "throughput", "latency"}:
        raise ValueError("invalid provider sort")
    if plan.get("provider_only", "") not in {"", "deepinfra", "parasail", "venice"}:
        raise ValueError("unsupported fixed provider")
    if plan.get("provider_only") and (
        plan["provider"] != "openrouter" or plan.get("compare_routing")
    ):
        raise ValueError(
            "fixed provider requires OpenRouter without routing comparison"
        )
    if plan.get("provider_sort") or plan.get("compare_routing"):
        if (
            plan["provider"] != "openrouter"
            or not plan["model"]
            or ":" in plan["model"]
        ):
            raise ValueError(
                "provider sorting requires OpenRouter and a base model "
                "without routing suffix"
            )
    if plan.get("compare_routing") and (
        plan["compare"]
        or plan.get("compare_concurrency")
        or plan.get("compare_arrival")
        or len(plan["streams"]) != 1
        or plan["soak_seconds"]
        or plan["repeats"] != 2
    ):
        raise ValueError(
            "routing comparison requires one stream count, repeats=2, "
            "soak=0 and no other comparison"
        )
    if plan.get("arrival_mode", "aligned") not in {"aligned", "staggered"}:
        raise ValueError("invalid arrival mode")
    if plan.get("compare_arrival") and (
        plan["compare"]
        or plan.get("compare_concurrency")
        or len(plan["streams"]) != 1
        or plan["streams"][0] < 2
        or plan["soak_seconds"]
        or plan["repeats"] != 2
    ):
        raise ValueError(
            "arrival comparison requires one multistream level, "
            "repeats=2, soak=0 and no other comparison"
        )
    if plan.get("event_json_mode", "prompt") not in {"prompt", "schema"}:
        raise ValueError("invalid event JSON mode")
    if plan.get("event_json_mode") == "schema" and plan["provider"] != "openrouter":
        raise ValueError("this schema experiment is for OpenRouter only")
    if plan.get("compare_concurrency") and (
        plan["compare"]
        or plan["streams"] != [1]
        or plan["soak_seconds"]
        or plan["repeats"] != 2
        or plan["global_limit"] < 4
    ):
        raise ValueError(
            "concurrency comparison requires streams=[1], repeats=2, "
            "soak=0, global_limit>=4 and no --compare"
        )
    if not isinstance(plan["model"], str) or len(plan["model"]) > 200:
        raise ValueError("invalid model")
    if plan["soak_seconds"] and 12 not in plan["streams"]:
        raise ValueError("soak requires a 12-stream ramp")
    if 0 < plan["soak_seconds"] < 30:
        raise ValueError("soak must be disabled (0) or at least 30 seconds")
    return plan


def make_plan(**overrides):
    return validate(
        {
            "version": 1,
            "provider": "openrouter",
            "model": "",
            "simulate": False,
            "streams": [1, 4, 8, 12],
            "duration": 30,
            "repeats": 2,
            "soak_seconds": 900,
            "compare": False,
            "compare_concurrency": False,
            "compare_arrival": False,
            "compare_routing": False,
            "provider_sort": "",
            "provider_only": "",
            "arrival_mode": "aligned",
            "event_json_mode": "prompt",
            "global_limit": 16,
            "per_stream_limit": 2,
            "request_rate": 20,
            "queue_timeout": 1,
            "observation_fps": 1.25,
            "deadline": 3,
            "min_coverage": 0.95,
            "max_error_fraction": 0.01,
            "max_uncertain_fraction": 0.10,
            **overrides,
        }
    )


def ramp_jobs(plan):
    if plan.get("compare_routing"):
        return [
            {
                "id": f"routing-{mode}-repeat-{repeat}",
                "streams": plan["streams"][0],
                "repeat": repeat,
                "variant": "shared",
                "duration": plan["duration"],
                "phase": "routing_ab",
                "provider_sort": mode,
            }
            for repeat, modes in [
                (1, ("throughput", "latency")),
                (2, ("latency", "throughput")),
            ]
            for mode in modes
        ]
    if plan.get("compare_arrival"):
        return [
            {
                "id": f"arrival-{mode}-repeat-{repeat}",
                "streams": plan["streams"][0],
                "repeat": repeat,
                "variant": "shared",
                "duration": plan["duration"],
                "phase": "arrival_ab",
                "arrival_mode": mode,
            }
            for repeat, modes in [
                (1, ("aligned", "staggered")),
                (2, ("staggered", "aligned")),
            ]
            for mode in modes
        ]
    if plan.get("compare_concurrency"):
        return [
            {
                "id": f"concurrency-{limit}-repeat-{repeat}",
                "streams": 1,
                "repeat": repeat,
                "variant": "shared",
                "duration": plan["duration"],
                "phase": "concurrency_ab",
                "per_stream_limit": limit,
            }
            for repeat, limits in [(1, (2, 4)), (2, (4, 2))]
            for limit in limits
        ]
    jobs = []
    for streams in plan["streams"]:
        for repeat in range(1, plan["repeats"] + 1):
            variants = ["unlimited", "shared"] if plan["compare"] else ["shared"]
            if repeat % 2 == 0:
                variants.reverse()
            for variant in variants:
                jobs.append(
                    {
                        "id": f"ramp-{streams}-{repeat}-{variant}",
                        "streams": streams,
                        "repeat": repeat,
                        "variant": variant,
                        "duration": plan["duration"],
                        "phase": "ramp",
                    }
                )
    return jobs


def request_budget(plan):
    # Includes one isolated real-response probe; transport HEADs
    # are not model inference. Soak runs only when the 12-stream ramp passes.
    ramp = sum(
        j["streams"] * math.ceil(j["duration"] * plan["observation_fps"])
        for j in ramp_jobs(plan)
    )
    return {
        "probe_upper": 1,
        "ramp_upper": ramp,
        "conditional_soak_upper": math.ceil(
            12 * plan["soak_seconds"] * plan["observation_fps"]
        ),
        "paid_requests": not plan["simulate"],
    }


def gate(stage, plan):
    reasons = []
    if len(stage["per_stream"]) != stage["streams"] or not stage["per_stream"]:
        reasons.append("missing_stream_results")
    if not stage["all_drained"]:
        reasons.append("queue_not_drained")
    if not stage["input_timing_valid"] or stage.get("result_records_missing", 0):
        reasons.append("invalid_input_or_result_timing")
    for camera, row in enumerate(stage["per_stream"], 1):
        expected = row["expected_observations"]
        if not expected or (row["on_time_coverage"] or 0) < plan["min_coverage"]:
            reasons.append(f"camera{camera}:low_on_time_coverage")
        if sum(row["errors"].values()) / max(1, expected) > plan["max_error_fraction"]:
            reasons.append(f"camera{camera}:request_errors")
        uncertain = row["uncertain_judgment_fraction"]
        if uncertain is None or uncertain > plan["max_uncertain_fraction"]:
            reasons.append(f"camera{camera}:uncertain_judgments")
    return {"passed": not reasons, "reasons": reasons}


def availability_gate(stage):
    """A complete response proves availability, not throughput or the 3 s SLA."""
    valid = (
        stage["streams"] == 1
        and len(stage["per_stream"]) == 1
        and stage["all_drained"]
        and stage["input_timing_valid"]
        and not stage.get("result_records_missing", 0)
        and stage["per_stream"][0]["complete_valid_results"] >= 1
    )
    return {"passed": valid, "reasons": [] if valid else ["no_valid_isolated_probe"]}


def soak_allowed(completed, plan):
    twelve = [
        j
        for j in completed
        if j["phase"] == "ramp" and j["streams"] == 12 and j["variant"] == "shared"
    ]
    return len(twelve) == plan["repeats"] and all(j["gate"]["passed"] for j in twelve)


def concurrency_comparison(completed, plan):
    """Keep partial arms visible; never hide a failed repeat behind a mean."""
    if not plan.get("compare_concurrency"):
        return []
    result = []
    for limit in (2, 4):
        jobs = [
            j
            for j in completed
            if j["phase"] == "concurrency_ab" and j["per_stream_limit"] == limit
        ]
        rows = [r for j in jobs for r in j["result"]["per_stream"]]
        planned = sum(r["expected_observations"] for r in rows)
        timely = sum(r["on_time_valid_results"] for r in rows)
        errors = {}
        for row in rows:
            for code, count in row["errors"].items():
                errors[code] = errors.get(code, 0) + count
        complete = len(jobs) == plan["repeats"] and {j["repeat"] for j in jobs} == {
            1,
            2,
        }
        result.append(
            {
                "per_stream_limit": limit,
                "completed_trials": len(jobs),
                "required_trials": plan["repeats"],
                "complete": complete,
                "all_trials_passed": complete
                and all(j["gate"]["passed"] for j in jobs),
                "expected_observations": planned,
                "complete_valid_results": sum(
                    r["complete_valid_results"] for r in rows
                ),
                "on_time_valid_results": timely,
                "on_time_coverage": timely / planned if planned else None,
                "errors": errors,
                "trial_ids": [j["id"] for j in jobs],
            }
        )
    return result
