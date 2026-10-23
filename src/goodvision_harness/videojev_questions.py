"""Compile fixed native choice messages once; fill only source times per window."""

import json
from collections import OrderedDict
from dataclasses import dataclass


_SOURCE_TIMES = "{source_times}"


@dataclass(eq=False)
class _Prefix:
    ids: object
    tokens: tuple
    device_ids: object = None
    embeddings: object = None
    state: object = None

    @property
    def length(self):
        return len(self.tokens)


@dataclass(eq=False)
class _TimeSlot:
    before: str
    after: str


class _Text:
    def __init__(self, processor, text, slots, *, first=False, spatial=1):
        tokenizer = processor.tokenizer
        left, marker, right = text.rpartition(_SOURCE_TIMES)
        self.slot = None
        if marker:
            # Cache whole pre-tokenizer units. Keep the adjacent word units in
            # the dynamic span so JSON punctuation/whitespace is not split at
            # an arbitrary BPE boundary when the source times change.
            sample = left + "[]" + right
            start, end = len(left), len(left) + 2
            units = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sample)
            before = [(a, b) for _, (a, b) in units if b <= start
                      and any(character.isalpha() for character in sample[a:b])]
            after = [(a, b) for _, (a, b) in units if a >= end
                     and any(character.isalpha() for character in sample[a:b])]
            lower = before[-1][0] if before else 0
            upper = after[0][1] if after else len(sample)
            dynamic = (sample[lower:start], sample[end:upper])
            self.slot = next((slot for slot in slots
                              if (slot.before, slot.after) == dynamic), None)
            if self.slot is None:
                self.slot = _TimeSlot(*dynamic)
                slots.append(self.slot)
            head, tail = sample[:lower], sample[upper:]
        else:
            head, tail = text, ""
        if first:
            # Match the native processor's first-message tokenization, then
            # expand each temporal placeholder to its actual spatial grid.
            ids = processor(text=[head], return_tensors="pt")["input_ids"][0].tolist()
        else:
            ids = tokenizer.encode(head, add_special_tokens=False)
        self.before = tuple(value for token in ids
                            for value in [token] * (
                                spatial if token == processor.video_token_id else 1
                            ))
        self.after = tuple(tokenizer.encode(tail, add_special_tokens=False))

    def length(self, dynamic):
        return len(self.before) + len(self.after) + (
            len(dynamic[self.slot]) if self.slot is not None else 0
        )

    def tokens(self, dynamic):
        middle = dynamic[self.slot] if self.slot is not None else ()
        return [*self.before, *middle, *self.after]


class CompiledQuestions:
    """Own one fixed scene-choice template and a small native geometry cache."""

    def __init__(self, processor, torch, rows):
        self.processor, self.torch = processor, torch
        self.prefix = None
        self.rows = tuple({
            **row, "query": "[]".join(row["query_parts"]),
        } for row in rows)
        # Preserve the native suffix contract once when binding the fixed text,
        # rather than rescanning rules and options in every candidate group.
        for row in self.rows:
            for text in (row["query"], *row["options"]):
                if not text.strip() or "<|" in text:
                    raise ValueError("questions/options must be text without control tokens")
        self.geometry = OrderedDict()
        self.decide_id = processor.tokenizer.convert_tokens_to_ids("<|decide|>")
        self.option_id = processor.tokenizer.convert_tokens_to_ids("<|option_end|>")
        divided = [row["query"].rpartition("\nFor ") for row in self.rows]
        self.shared = len(rows) > 1 and bool(divided[0][0]) and all(
            marker and head == divided[0][0] for head, marker, _ in divided
        )
        self.following = tuple(
            {**row, "query": marker + question}
            for row, (_, marker, question) in zip(self.rows, divided)
        ) if self.shared else self.rows
        policies = [row["query"].partition("\nEvidence metadata: ") for row in self.rows]
        policy = policies[0][0]
        self.prefixed = tuple({
            **row, "query_parts": (row["query_parts"][0][len(policy):],
                                    row["query_parts"][1]),
        } for row in self.rows) if self.shared and policy and all(
            marker and head == policy for head, marker, _ in policies
        ) else self.rows

    def _full_prefix(self):
        if self.prefix is not None:
            return self.prefix
        fixed = []
        for row in self.rows:
            policy, marker, evidence = row["query"].partition("\nEvidence metadata: ")
            query = "[current window source times]".join(row["query_parts"])
            if marker:
                try:
                    metadata, end = json.JSONDecoder().raw_decode(evidence)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(metadata, dict):
                        metadata.pop("source_times", None)
                        query = policy + (
                            marker + json.dumps(metadata, allow_nan=False) if metadata else ""
                        ) + evidence[end:]
            fixed.append(query)
        divided = [query.rpartition("\nFor ") for query in fixed]
        common = divided[0][0] if all(
            marker and head == divided[0][0] for head, marker, _ in divided
        ) else ""
        text = common + "\nTask: single-choice video question answering\n"
        for index, (row, query, (_, marker, question)) in enumerate(
            zip(self.rows, fixed, divided)
        ):
            text += f"\nQuestion {index + 1}: {marker + question if common else query}\nOptions:\n"
            text += "".join(option + "\n" for option in row["options"])
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": text}]}],
            tokenize=False, add_generation_prompt=False,
        )
        tokens = tuple(self.processor.tokenizer.encode(text, add_special_tokens=False))
        self.prefix = _Prefix(self.torch.tensor([tokens], dtype=self.torch.long), tokens)
        return self.prefix

    def _messages(self, record):
        from transformers.video_utils import VideoMetadata
        import numpy as np

        key = (record["frames"], record["fps"], *record["grid_thw"])
        if key in self.geometry:
            self.geometry.move_to_end(key)
            return self.geometry[key]
        processor, slots = self.processor, []
        grid = record["grid_thw"]
        replacement = processor.replace_video_token({
            "video_grid_thw": np.asarray([[grid[0], 2, 2]]),
            "video_metadata": [VideoMetadata(
                total_num_frames=record["frames"], fps=record["fps"],
                frames_indices=list(range(record["frames"])),
            )],
        }, video_idx=0)

        def message(row, video=False, dynamic=True):
            query = _SOURCE_TIMES.join(row["query_parts"]) if dynamic else row["query"]
            text = "\nTask: single-choice video question answering\nQuestion: " + query
            text += "\nOptions:\n" + "".join(
                option + "<|option_end|>\n" for option in row["options"]
            ) + "<|decide|>"
            content = [{"type": "video", "video": "provided_in_memory"}] if video else []
            content.append({"type": "text", "text": text})
            text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=False,
            )
            if video:
                text = processor.get_text_with_replacements([text], [], [replacement], [])[0][0]
            return _Text(processor, text, slots, first=video,
                         spatial=grid[1] * grid[2] // 4)

        original_first = tuple(message(row, True) for row in self.rows)
        original_next = tuple(message(row) for row in self.rows)
        shared_next = tuple(message(row, dynamic=False) for row in self.following)
        prefix_first = tuple(message(row, True) for row in self.prefixed)
        value = original_first, original_next, shared_next, prefix_first, slots
        self.geometry[key] = value
        while len(self.geometry) > 8:
            self.geometry.popitem(last=False)
        return value

    def groups(self, record, frame_times, context, *, prefix_cache=False):
        first, following, shared, prefixed, slots = self._messages(record)
        times = json.dumps(frame_times, allow_nan=False)
        tokenizer = self.processor.tokenizer
        dynamic = {slot: tokenizer.encode(slot.before + times + slot.after,
                                           add_special_tokens=False) for slot in slots}
        prefix = self._full_prefix() if prefix_cache else None
        if prefix is not None and any(
            (prefix.length + part.length(dynamic) + 63) // 64 * 64 > context
            for part in prefixed
        ):
            # Keep every original question when a full prefix cannot fit.
            prefix = None
        if prefix is not None:
            first = prefixed
            following = shared if self.shared else following
        start = 0
        while start < len(self.rows):
            # Preserve native repeated-layout group boundaries, but determine
            # them by lengths instead of repeatedly rebuilding each candidate.
            end = start + 1
            length = first[start].length(dynamic) + (0 if prefix is None else prefix.length)
            if (length + 63) // 64 * 64 > context:
                raise ValueError("a complete native choice exceeds the context")
            while end < len(self.rows):
                candidate = length + following[end].length(dynamic)
                if (candidate + 63) // 64 * 64 > context:
                    break
                length, end = candidate, end + 1
            selected = [first[start], *following[start + 1:end]]
            if prefix is None and self.shared and end - start > 1:
                candidate = [first[start], *shared[start + 1:end]]
                candidate_length = sum(part.length(dynamic) for part in candidate)
                if (candidate_length + 63) // 64 * 64 <= context:
                    selected = candidate
            values = [] if prefix is None else list(prefix.tokens)
            for part in selected:
                values.extend(part.tokens(dynamic))
            pairs = [index for index, token in enumerate(values) if token == self.option_id]
            queries = [index for index, token in enumerate(values) if token == self.decide_id]
            torch = self.torch
            ids = torch.tensor([values], dtype=torch.long)
            count = end - start
            batch = {
                "input_ids": ids,
                "mm_token_type_ids": (ids == self.processor.video_token_id).long() * 2,
                "video_grid_thw": torch.tensor([record["grid_thw"]]),
                "pair_indices": torch.tensor(pairs, dtype=torch.long).reshape(count, 3),
                "query_indices": torch.tensor(queries, dtype=torch.long),
                "score_mask": torch.ones((count, 3), dtype=torch.bool),
                "decision_tasks": torch.ones(count, dtype=torch.long),
            }
            yield batch, self.rows[start:end], prefix
            start = end
