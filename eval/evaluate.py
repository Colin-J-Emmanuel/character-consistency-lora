"""Run the full character-consistency evaluation.

Example:
    python eval/evaluate.py \
        --ref_dir data/my_character \
        --gen_dir outputs/samples \
        --prompts_file scripts/prompts.txt \
        --out outputs/eval_results.json

Optionally pass --baseline_gen_dir to compare a base-SDXL run against the LoRA run
(the strongest way to show your fine-tune actually moved identity metrics).
"""
import argparse
import json
import os
from glob import glob

from PIL import Image

from metrics import CLIPScorer, DINOScorer, clip_t, image_identity, self_consistency


IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def load_images(folder: str) -> list[Image.Image]:
    paths = []
    for ext in IMG_EXT:
        paths.extend(sorted(glob(os.path.join(folder, ext))))
    # Skip contact-sheet grids so they don't pollute identity scores.
    paths = [p for p in paths if "grid" not in os.path.basename(p).lower()]
    return [Image.open(p).convert("RGB") for p in paths]


def load_prompts(prompts_file: str, n: int) -> list[str]:
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < n:
        prompts += [""] * (n - len(prompts))
    return prompts[:n]


def evaluate(ref_dir, gen_dir, prompts_file):
    ref = load_images(ref_dir)
    gen = load_images(gen_dir)
    if not gen:
        raise SystemExit(f"No generated images found in {gen_dir}")
    prompts = load_prompts(prompts_file, len(gen))

    clip = CLIPScorer()
    dino = DINOScorer()

    results = {
        "n_reference": len(ref),
        "n_generated": len(gen),
        "clip_t_prompt_fidelity": round(clip_t(clip, gen, prompts), 4),
        "clip_i_identity": round(image_identity(clip, gen, ref), 4) if ref else None,
        "dino_identity": round(image_identity(dino, gen, ref), 4) if ref else None,
        "self_consistency_clip": round(self_consistency(clip, gen), 4),
        "self_consistency_dino": round(self_consistency(dino, gen), 4),
    }
    return results


def print_table(results: dict):
    rows = [
        ("CLIP-T (prompt fidelity)", results["clip_t_prompt_fidelity"]),
        ("CLIP-I (identity)", results["clip_i_identity"]),
        ("DINO (identity)", results["dino_identity"]),
        ("Self-consistency (CLIP)", results["self_consistency_clip"]),
        ("Self-consistency (DINO)", results["self_consistency_dino"]),
    ]
    print("\n=== Character-consistency eval ===")
    print(f"reference imgs: {results['n_reference']}  |  generated imgs: {results['n_generated']}")
    for name, val in rows:
        val_str = "n/a" if val is None else f"{val:.4f}"
        print(f"  {name:<28} {val_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", required=True, help="Reference character photos")
    p.add_argument("--gen_dir", required=True, help="Generated images to score")
    p.add_argument("--prompts_file", default="scripts/prompts.txt")
    p.add_argument("--baseline_gen_dir", default=None,
                   help="Optional: base-SDXL generations for A/B comparison")
    p.add_argument("--out", default="outputs/eval_results.json")
    args = p.parse_args()

    report = {"lora": evaluate(args.ref_dir, args.gen_dir, args.prompts_file)}
    print("\n[LoRA fine-tuned]")
    print_table(report["lora"])

    if args.baseline_gen_dir:
        report["baseline"] = evaluate(args.ref_dir, args.baseline_gen_dir, args.prompts_file)
        print("\n[Base SDXL]")
        print_table(report["baseline"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
