#!/usr/bin/env python3
"""Pre-tokenize staged DPO pairs for one model, with that model's own tokenizer and bundled chat template.

Models of the same family can differ in special-token ids (Prelude-SFT-v2 has <|im_start|> = 262145, our 256K line
has 3), so pairs are tokenized per model and never shared. Default mode (enable_thinking=False), exactly as
sample.py rendered the prompts. With --mode concat, a response is the generation prompt followed by the reply's tokens
and the template's end-of-turn suffix, which is what the model sees at inference when its template's generation prompt
differs from a rendered history turn. The ids follow TRL's DPOTrainer (prompt with add_generation_prompt=True, completion =
render(prompt + response)[len(prompt):]). A pair is dropped if a response is empty, the render fails, the prompt is
not a token prefix of prompt + response, a sequence exceeds --max-length, or a response contains a chat-tag token
(think and tool delimiters, looked up in this tokenizer).

  python pretok_pairs.py --staged runs/mix-x/staged --tokenizer <model dir> --out runs/mix-x/pretok
Output: {split}-0000.parquet with id, source, language, prompt_ids, chosen_ids, rejected_ids, and manifest.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TAGS = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<tools>", "</tools>"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="model dir with tokenizer.json and chat_template.jinja")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--mode", choices=["render", "concat"], default="render",
                    help="render: TRL's way (prompt + response rendered together); concat: generation prompt + reply "
                         "tokens + end suffix, for templates whose generation prompt differs from a rendered history turn "
                         "(Prelude-SFT-v2 prefills an empty think block that a rendered history turn does not have)")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    template = a.tokenizer / "chat_template.jinja"
    tag_ids = {i for t in TAGS if (i := tok.convert_tokens_to_ids(t)) is not None and i != tok.unk_token_id}

    def ids(messages: list[dict], gen: bool) -> list[int]:
        out = tok.apply_chat_template(messages, add_generation_prompt=gen, tokenize=True, enable_thinking=False,
                                      return_dict=False)
        return list(out[0] if out and isinstance(out[0], list) else out)

    # The text the template puts after an assistant message ("<|im_end|>\n"), found by rendering a probe turn.
    probe = "PROBE_RESPONSE_TEXT"
    rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": probe}],
                                       tokenize=False, enable_thinking=False)
    suffix_ids = tok.encode(rendered[rendered.rindex(probe) + len(probe):], add_special_tokens=False)

    def response_ids(messages: list[dict]) -> list[int]:
        """concat mode: the generation prompt (as at inference, e.g. with an empty think block) + the reply + suffix."""
        assert len(messages) == 1 and messages[0]["role"] == "assistant"
        return tok.encode(messages[0]["content"].strip(), add_special_tokens=False) + suffix_ids

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"kind": "oellm_dpo_v1_pretokenized", "tokenizer": str(a.tokenizer),  # the kind dpo_train_pretok.py accepts
                "tokenizer_json_sha256": sha(a.tokenizer / "tokenizer.json"),
                "template_sha256": sha(template) if template.exists() else None, "enable_thinking": False,
                "max_length": a.max_length, "mode": a.mode, "suffix_ids": suffix_ids,
                "response_tag_ids": sorted(tag_ids), "splits": {}}
    for split in ("train", "validation"):
        cols = {k: [] for k in ("id", "source", "language", "prompt_ids", "chosen_ids", "rejected_ids")}
        stats: dict[str, Counter] = {}
        lengths = []
        for f in sorted(a.staged.glob(f"*-{split}.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    key = row["source"] if row["source"].startswith("oellm-natural") else row["source"].split(":")[0]
                    c = stats.setdefault(key, Counter())
                    c["staged"] += 1
                    reason = None
                    if not all((m.get("content") or "").strip() for side in ("chosen", "rejected") for m in row[side]):
                        reason = "empty_response"
                    else:
                        try:
                            p = ids(row["prompt"], True)
                            if a.mode == "render":
                                fc = ids(row["prompt"] + row["chosen"], False)
                                fr = ids(row["prompt"] + row["rejected"], False)
                            else:
                                fc, fr = p + response_ids(row["chosen"]), p + response_ids(row["rejected"])
                        except Exception:  # noqa: BLE001 - template errors, bad roles
                            reason = "render_error"
                        else:
                            if fc[: len(p)] != p or fr[: len(p)] != p:
                                reason = "prefix_mismatch"
                            elif len(fc) > a.max_length or len(fr) > a.max_length:
                                reason = "over_max_length"
                            elif tag_ids & (set(fc[len(p):]) | set(fr[len(p):])):
                                reason = "chat_tag_in_response"
                    if reason:
                        c[f"dropped_{reason}"] += 1
                        continue
                    c["kept"] += 1
                    lengths.append(len(fc) if len(fc) > len(fr) else len(fr))
                    for k, v in (("id", row["id"]), ("source", row["source"]), ("language", row.get("language", "")),
                                 ("prompt_ids", p), ("chosen_ids", fc[len(p):]), ("rejected_ids", fr[len(p):])):
                        cols[k].append(v)
        table = pa.table({**{k: cols[k] for k in ("id", "source", "language")},
                          **{k: pa.array(cols[k], type=pa.list_(pa.int32())) for k in ("prompt_ids", "chosen_ids", "rejected_ids")}})
        pq.write_table(table, a.out / f"{split}-0000.parquet", compression="zstd")
        lengths.sort()
        manifest["splits"][split] = {"by_source": {s: dict(v) for s, v in sorted(stats.items())},
                                     "kept": len(cols["id"]),
                                     "length_p50": lengths[len(lengths) // 2] if lengths else None,
                                     "length_p95": lengths[int(len(lengths) * 0.95)] if lengths else None}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
