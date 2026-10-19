"""Local token budgeting for the frozen two-message Qwen final-answer route."""
from functools import lru_cache
import json
from pathlib import Path

from ..benchmark_evidence_encoding import expand_repeated_evidence
from .summary_encoding import compact_window_evidence
from .summary_input import message_bytes, prepare_summary_messages


@lru_cache(maxsize=2)
def _tokenizer(path):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(path)))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def context_usage(messages, *, model, parameters, budget):
    """Count text using Qwen's two-role chat template, reserving output and margin."""
    if (model != "qwen/qwen3-vl-8b-instruct" or budget["model"] != model
            or len(messages) != 2 or [row.get("role") for row in messages] != ["system", "user"]
            or any(not isinstance(row.get("content"), str) for row in messages)
            or set(parameters) - {"temperature", "max_tokens"}):
        raise ValueError("Token budgeting requires the frozen Qwen two-message text route")
    context, margin, output = (budget["context_tokens"], budget["margin_tokens"], parameters["max_tokens"])
    if (any(type(value) is not int or value < 1 for value in (context, margin, output))
            or output + margin >= context):
        raise ValueError("Explicit positive context, output and safety budgets required")
    rendered = "".join("<|im_start|>" + row["role"] + "\n" + row["content"]
                       + "<|im_end|>\n" for row in messages) + "<|im_start|>assistant\n"
    count = len(_tokenizer(budget["tokenizer_path"]).encode(rendered, add_special_tokens=False).ids)
    return {"prompt_tokens": count, "context_tokens": context, "output_tokens": output,
            "margin_tokens": margin, "fits": count + output + margin <= context,
            "tokenizer_source": budget["tokenizer_source"],
            "count_method": "local_qwen_text_chat_template_not_provider_receipt"}


def prepare_context_messages(messages, *, model, parameters, max_input_bytes, budget):
    usage = context_usage(messages, model=model, parameters=parameters, budget=budget)
    original_size = len(message_bytes(messages))
    if original_size <= max_input_bytes and usage["fits"]:
        return messages, {"method": "lossless_context_input_v1", "encoding": "unchanged",
                          "input_bytes": original_size, "usage": usage, "budget": budget,
                          "records_omitted": 0, "candidates": []}
    payload = json.loads(messages[1]["content"])
    evidence = expand_repeated_evidence(payload["evidence"])
    try:
        window_evidence = compact_window_evidence(evidence)
    except ValueError as exc:
        return messages, {"method": "lossless_context_input_v1", "encoding": "previous",
                          "input_bytes": original_size, "usage": usage, "budget": budget,
                          "records_omitted": 0, "candidates": [
                              {"encoding": "window_tables", "status": "retained_previous",
                               "reason": str(exc)}]}
    candidate = [messages[0], {**messages[1], "content": json.dumps(
        {**payload, "evidence": window_evidence}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)}]
    candidates = []
    selected, selected_usage, selected_size, encoding = messages, usage, original_size, "previous"
    # Keep the legacy representation for paid identities; this function handles
    # only never-sent inputs. Compare the new table with its existing schema codec.
    schema, _ = prepare_summary_messages(candidate, max_input_bytes=1)
    for name, value in (("window_tables", candidate), ("window_tables_schema", schema)):
        size = len(message_bytes(value))
        measured = context_usage(value, model=model, parameters=parameters, budget=budget)
        candidates.append({"encoding": name, "input_bytes": size, "usage": measured})
        rank = (not (size <= max_input_bytes and measured["fits"]), measured["prompt_tokens"], size)
        previous_rank = (not (selected_size <= max_input_bytes and selected_usage["fits"]),
                         selected_usage["prompt_tokens"], selected_size)
        if rank < previous_rank:
            selected, selected_usage, selected_size, encoding = value, measured, size, name
    return selected, {"method": "lossless_context_input_v1", "encoding": encoding,
                      "input_bytes": selected_size, "usage": selected_usage, "budget": budget,
                      "records_omitted": 0, "question_unchanged": True, "truncated": False,
                      "candidates": candidates}
