#!/usr/bin/env python3
"""Figure for a benchmark run: what each model does with made-up subjects, obscure real people and famous people.

  python plot.py ../results/baseline-2026-11-13.json --out ../results/baseline-2026-11-13.png [--order "E10 / default" ...]

Reads the JSON written by `score.py --json` (keys "<model> / <mode>").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

RED, AMBER, GREEN, GREY, INK, MUTED = "#d1495b", "#edae49", "#2e9e6b", "#d9d9d9", "#1f2933", "#6b7785"


def font() -> str:
    for name in ("Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            return name
    return "sans-serif"


def segments(m: dict) -> list[list[tuple[float, str, str]]]:
    made_fab, made_good, made_poor = m["fabrication"], m["good_abstention"], m["abstention_poor"]
    obs_ok, obs_wrong, obs_abs = m["obscure_acc"], m["obscure_wrong"], m["obscure_abstain"]
    known_ok, known_ref = m["known_acc"], m["over_refusal"]
    rest = lambda *xs: max(0.0, 100 - sum(xs))  # noqa: E731  OTHER (empty or cut-off answers)
    return [
        [(made_fab, RED, "invents an answer"), (made_poor, AMBER, "refuses unhelpfully"),
         (made_good, GREEN, "honest \"I don't know\""), (rest(made_fab, made_poor, made_good), GREY, "other")],
        [(obs_ok, GREEN, "correct"), (obs_wrong, RED, "wrong"), (obs_abs, AMBER, "abstains"),
         (rest(obs_ok, obs_wrong, obs_abs), GREY, "other")],
        [(known_ok, GREEN, "correct"), (rest(known_ok, known_ref), RED, "wrong"), (known_ref, AMBER, "refuses")],
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", nargs="+", help="rows, top to bottom (default: as in the report)")
    ap.add_argument("--highlight", default="OpenEuroLLM", help="prefix of rows drawn in bold")
    ap.add_argument("--title", default="How often do models make things up?")
    ap.add_argument("--subtitle", default="500 natural questions in 10 European languages, no answer format, "
                                          "closed-book; judged by gemma-4-31B-it")
    a = ap.parse_args()
    rep = json.loads(a.report.read_text())
    rows = [k for k in (a.order or rep) if " / " in k and k.count(" / ") == 1]
    plt.rcParams.update({"font.family": font(), "font.size": 10.5, "axes.edgecolor": "none",
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})
    height = 2.1 + 0.5 * len(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, height), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.08})
    fig.subplots_adjust(top=1 - 1.45 / height, bottom=0.45 / height, left=0.17, right=0.995)
    titles = ["Made-up people, novels, films, firms, places", "Real but obscure people (1-3 Wikipedia editions)",
              "Well-known people (60+ editions)"]
    for p, (ax, title) in enumerate(zip(axes, titles)):
        for y, key in enumerate(rows):
            left = 0.0
            for value, color, label in segments(rep[key])[p]:
                ax.barh(y, value, left=left, color=color, height=0.68, edgecolor="white", linewidth=1.2)
                if value >= 9:
                    ax.text(left + value / 2, y, f"{value:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if color in (RED, GREEN) else INK, fontweight="bold")
                left += value
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=22, fontweight="semibold")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100 %"])
        ax.tick_params(axis="both", length=0)
        ax.grid(axis="x", color="#eef0f2", linewidth=0.8)
        ax.set_axisbelow(True)
        seen, handles = set(), []
        for _, color, label in segments(rep[rows[0]])[p]:
            if label not in seen and label != "other":
                seen.add(label)
                handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(handles), frameon=False,
                  fontsize=9, handlelength=1, handleheight=1, columnspacing=1.0, borderaxespad=0.15)
    axes[0].axvline(10, color=INK, linewidth=1.1, linestyle=(0, (3, 2)))
    axes[0].text(11, -0.62, "target < 10 %", fontsize=8.5, color=INK, va="bottom")
    groups = [r.split(" / ")[0].split()[0] for r in rows]  # separator where the model family changes
    for ax in axes:
        for y in range(1, len(rows)):
            if (groups[y] == a.highlight) != (groups[y - 1] == a.highlight):
                ax.axhline(y - 0.5, color=MUTED, linewidth=0.8)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(rows)
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text().startswith(a.highlight):
            lbl.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.text(0.005, 1 - 0.25 / height, a.title, ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.005, 1 - 0.62 / height, a.subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print("->", a.out)


if __name__ == "__main__":
    main()
