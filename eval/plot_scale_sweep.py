#!/usr/bin/env python
"""
Plot the LoRA scale sweep from the committed evaluation files.

Reads each run's eval_sface.json (identity) and eval.json (scene fidelity) and
draws the two curves side by side, so the figure is regenerated from the
recorded numbers rather than maintained by hand.

Usage:
    python eval/plot_scale_sweep.py
    python eval/plot_scale_sweep.py --out results/scale_sweep.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LoRA scale -> results directory. 0.0 is the text-only baseline.
POINTS = [
    (0.0, "results/baseline"),
    (0.4, "results/scale_04"),
    (0.7, "results/scale_07"),
    (1.0, "results/lora_desc"),
]

ARMS = [
    ("prompt_only", "varied seed, CFG 7"),
    ("fixed_seed", "fixed seed, CFG 7"),
    ("high_cfg", "fixed seed, CFG 12"),
]


def load(path, key):
    with open(path) as f:
        return json.load(f)["arms"], key


def series(metric_file, metric_key):
    """Return {arm: [value per scale]} for one metric, skipping missing runs."""
    out = {arm: [] for arm, _ in ARMS}
    scales = []
    for scale, folder in POINTS:
        path = os.path.join(folder, metric_file)
        if not os.path.exists(path):
            print(f"skipping {path}, not found")
            continue
        arms = json.load(open(path))["arms"]
        scales.append(scale)
        for arm, _ in ARMS:
            out[arm].append(arms[arm][metric_key])
    return scales, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scale_sweep.png")
    args = ap.parse_args()

    scales, identity = series("eval_sface.json", "sface_identity_mean")
    _, fidelity = series("eval.json", "clip_t_scene_fidelity_mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ["#2b6cb0", "#c05621", "#2f855a"]

    for (arm, label), color in zip(ARMS, colors):
        ax1.plot(scales, identity[arm], marker="o", color=color, label=f"{arm} ({label})")
        ax2.plot(scales, fidelity[arm], marker="o", color=color)

    ax1.set_title("Identity: SFace similarity")
    ax1.set_ylabel("mean pairwise similarity")
    ax1.set_ylim(0.45, 0.88)

    ax2.set_title("Scene fidelity: CLIP-T")
    ax2.set_ylabel("mean image-to-scene similarity")
    ax2.set_ylim(0.18, 0.26)

    for ax in (ax1, ax2):
        ax.set_xlabel("LoRA scale (0.0 = text only, no adapter)")
        ax.set_xticks(scales)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Adapter strength buys identity at little cost to scene fidelity", y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
