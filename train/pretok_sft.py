#!/usr/bin/env python3
"""Pre-tokenize SFT rows (messages) for one model with its own tokenizer and chat template, default mode.

Each assistant turn becomes one example: the template's generation prompt for the conversation so far (as the model
sees it at inference, e.g. with an empty think block), then the reply's tokens and the template's end-of-turn suffix.
Only the reply and suffix are supervised (assistant_masks = 1). This is the same construction as pretok_pairs.py
--mode concat, so SFT and DPO train exactly what the model generates in default mode.

  python pretok_sft.py --rows runs/x/warmstart --tokenizer <model dir> --out runs/x/warmstart/pretok
Reads sft-{train,validation}.jsonl.gz; writes {split}-0000.parquet (input_ids, assistant_masks, id, source) and
manifest.json. Examples longer than --max-length and replies containing chat-tag tokens are dropped.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tag_ids = {i for t in TAGS if (i := tok.convert_tokens_to_ids(t)) is not None and i != tok.unk_token_id}
    probe = "PROBE_RESPONSE_TEXT"
    rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": probe}],
                                       tokenize=False, enable_thinking=False)
    suffix = tok.encode(rendered[rendered.rindex(probe) + len(probe):], add_special_tokens=False)

    def prompt_ids(messages: list[dict]) -> list[int]:
        out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
                                      return_dict=False)
        return list(out["input_ids"] if isinstance(out, dict) else out)

    a.out.mkdir(parents=True, exist_ok=True)
    template = a.tokenizer / "chat_template.jinja"
    manifest = {"kind": "oellm_sft_pretokenized_per_model", "tokenizer": str(a.tokenizer),
                "tokenizer_json_sha256": hashlib.sha256((a.tokenizer / "tokenizer.json").read_bytes()).hexdigest(),
                "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest() if template.exists() else None,
                "enable_thinking": False, "suffix_ids": suffix, "reply_tag_ids": sorted(tag_ids),
                "max_length": a.max_length, "splits": {}}
    for split in ("train", "validation"):
        cols = {"id": [], "source": [], "input_ids": [], "assistant_masks": []}
        stats: dict[str, Counter] = {}
        with gzip.open(a.rows / f"sft-{split}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = row["source"].split(":")[0] if not row["source"].startswith("oellm-natural") else row["source"]
                c = stats.setdefault(key, Counter())
                msgs = row["messages"]
                for i, m in enumerate(msgs):
                    if m["role"] != "assistant":
                        continue
                    c["turns"] += 1
                    reply = tok.encode(m["content"].strip(), add_special_tokens=False) + suffix
                    prefix = prompt_ids(msgs[:i])
                    ids = prefix + reply
                    if len(ids) > a.max_length:
                        c["dropped_over_max_length"] += 1
                        continue
                    if tag_ids & set(reply):
                        c["dropped_chat_tag_in_reply"] += 1
                        continue
                    c["kept"] += 1
                    cols["id"].append(f"{row['id']}#{i}")
                    cols["source"].append(row["source"])
                    cols["input_ids"].append(ids)
                    cols["assistant_masks"].append([0] * len(prefix) + [1] * len(reply))
        pq.write_table(pa.table({"id": cols["id"], "source": cols["source"],
                                 "input_ids": pa.array(cols["input_ids"], type=pa.list_(pa.int32())),
                                 "assistant_masks": pa.array(cols["assistant_masks"], type=pa.list_(pa.int8()))}),
                       a.out / f"{split}-0000.parquet", compression="zstd")
        manifest["splits"][split] = {"examples": len(cols["id"]),
                                     "supervised_tokens": sum(sum(m) for m in cols["assistant_masks"]),
                                     "by_source": {s: dict(v) for s, v in sorted(stats.items())}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
