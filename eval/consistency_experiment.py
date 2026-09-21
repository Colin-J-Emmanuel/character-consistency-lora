#!/usr/bin/env python
"""
Character-consistency experiment harness.

One script, one set of metric definitions, so baseline and fine-tuned runs are
directly comparable. Same scenes, same seeds, same arms, both times.

Metrics
    CLIP-I self-consistency   mean pairwise cosine over CLIP image embeddings
    DINO self-consistency     mean pairwise cosine over DINOv2 CLS embeddings
    CLIP-T scene fidelity     cosine to the SCENE text only, character
                              description excluded, so identity and
                              prompt-following stay separable

Modes
    baseline      python consistency_experiment.py --out_dir results/baseline
    fine-tuned    python consistency_experiment.py --out_dir results/lora \
                      --lora_dir outputs/lora --character "sks woman"
    re-score      python consistency_experiment.py --out_dir results/lora --eval_only
    one folder    python consistency_experiment.py --eval_dir data/my_character
    face crops    python consistency_experiment.py --out_dir results/lora --eval_only --face_crop
    compare       python consistency_experiment.py --compare \
                      results/baseline/eval.json results/lora/eval.json
"""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Experiment definition. Keep these frozen across runs or comparisons are void.
# --------------------------------------------------------------------------

CHARACTER = (
    "a 28-year-old woman with short curly auburn hair, light freckles across "
    "the nose, green eyes, and a small scar above the left eyebrow"
)

SCENES = [
    "sitting at a table in a busy coffee shop",
    "standing on a city street at night in the rain",
    "hiking on a mountain trail in bright daylight",
    "in a spacesuit aboard a space station",
    "wearing medieval armor in a stone courtyard",
    "laughing at an outdoor summer picnic",
]

NEGATIVE = "blurry, lowres, deformed, extra limbs, watermark, text"

ARMS = {
    "prompt_only": {"fixed_seed": False, "guidance": 7.0},
    "fixed_seed": {"fixed_seed": True, "guidance": 7.0},
    "high_cfg": {"fixed_seed": True, "guidance": 12.0},
}

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FIXED_VAE = "madebyollin/sdxl-vae-fp16-fix"
CLIP_MODEL = "openai/clip-vit-large-patch14"
DINO_MODEL = "facebook/dinov2-base"


def place_on_gpu(pipe):
    """Keep the whole pipeline on GPU when it fits, offload when it does not.

    24GB cards (L4, A10) hold SDXL comfortably, and CPU offload would only
    add host-to-device traffic on every forward pass. 16GB cards (T4) need it.
    """
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 20:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()   # moved off the pipeline in diffusers 0.39


def _clip_text_embed(clip, **inputs):
    # transformers 5 returns a model output from get_text_features, not a
    # tensor. Calling the submodules directly is stable across versions and
    # makes the projection explicit: pooler_output alone is the pre-projection
    # hidden state (768 for text, 1024 for vision), not the shared CLIP space.
    out = clip.text_model(**inputs)
    pooled = out[1] if not isinstance(out, torch.Tensor) else out
    return clip.text_projection(pooled)


def _clip_image_embed(clip, **inputs):
    out = clip.vision_model(**inputs)
    pooled = out[1] if not isinstance(out, torch.Tensor) else out
    return clip.visual_projection(pooled)


def build_prompt(character: str, scene: str) -> str:
    return f"a photo of {character}, {scene}, natural lighting, 50mm portrait"


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def load_pipeline(lora_dir: str | None, lora_scale: float = 1.0):
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        sys.exit("Needs a GPU. Colab: Runtime > Change runtime type > T4 GPU.")

    # SDXL's stock VAE emits NaNs in fp16. This fixed VAE is why fp16 works.
    vae = AutoencoderKL.from_pretrained(FIXED_VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    if lora_dir:
        pipe.load_lora_weights(lora_dir, adapter_name="character")
        pipe.set_adapters(["character"], adapter_weights=[lora_scale])
        print(f"loaded LoRA from {lora_dir} at scale {lora_scale}")
    pipe.set_progress_bar_config(disable=True)
    place_on_gpu(pipe)
    return pipe


def generate(out_dir, character, steps, base_seed, size, lora_dir, lora_scale):
    pipe = load_pipeline(lora_dir, lora_scale)

    for arm_name, cfg in ARMS.items():
        arm_dir = os.path.join(out_dir, arm_name)
        os.makedirs(arm_dir, exist_ok=True)
        for i, scene in enumerate(SCENES):
            seed = base_seed if cfg["fixed_seed"] else base_seed + i
            image = pipe(
                prompt=build_prompt(character, scene),
                negative_prompt=NEGATIVE,
                num_inference_steps=steps,
                guidance_scale=cfg["guidance"],
                height=size,
                width=size,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
            image.save(os.path.join(arm_dir, f"{i:02d}.png"))
            print(f"[{arm_name}] scene {i} seed={seed} cfg={cfg['guidance']}")

    del pipe
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _load_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".webp")
    paths = sorted(p for p in os.listdir(folder) if p.lower().endswith(exts))
    return [Image.open(os.path.join(folder, p)).convert("RGB") for p in paths]


def _l2(x):
    return x / x.norm(dim=-1, keepdim=True)


def _mean_pairwise(emb):
    emb = _l2(emb)
    sims = [float(emb[i] @ emb[j]) for i, j in itertools.combinations(range(len(emb)), 2)]
    return float(np.mean(sims)), float(np.min(sims)), sims


def _face_crop(img, margin=0.3, min_frac=0.06):
    """Crop to the largest detected face, with margin, or return None.

    Whole-frame embeddings mix identity with scene, colour and composition.
    Scoring face crops isolates identity, which is what this project measures.
    OpenCV's Haar detector ships with opencv-python, so this adds no
    dependency. It is crude next to a learned detector plus ArcFace, but it is
    reliable on the mostly frontal faces these prompts produce.
    """
    import cv2

    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    h, w = gray.shape
    min_side = int(min(h, w) * min_frac)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side)
    )
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    mx, my = int(fw * margin), int(fh * margin)
    box = (max(0, x - mx), max(0, y - my), min(w, x + fw + mx), min(h, y + fh + my))
    return img.crop(box)


class Scorer:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.dino = AutoModel.from_pretrained(DINO_MODEL).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(DINO_MODEL)

    def embed(self, images):
        with torch.no_grad():
            ci = self.clip_proc(images=images, return_tensors="pt")
            clip_img = _clip_image_embed(self.clip,
                **{k: v.to(self.device) for k, v in ci.items()}
            )
            di = self.dino_proc(images=images, return_tensors="pt")
            # CLS token: the instance-level summary DINOv2 is trained to make
            # discriminative. This is why it is stricter than CLIP on identity.
            dino_img = self.dino(
                **{k: v.to(self.device) for k, v in di.items()}
            ).last_hidden_state[:, 0]
        return clip_img, dino_img

    def text(self, prompts):
        with torch.no_grad():
            t = self.clip_proc(
                text=prompts, return_tensors="pt", padding=True, truncation=True
            )
            return _l2(
                _clip_text_embed(self.clip,
                    **{k: v.to(self.device) for k, v in t.items()}
                )
            )

    def score_folder(self, folder, scene_emb=None, face_crop=False):
        images = _load_images(folder)
        missing = []
        if face_crop:
            # Scene fidelity only makes sense on whole frames.
            scene_emb = None
            crops = [_face_crop(im) for im in images]
            missing = [i for i, c in enumerate(crops) if c is None]
            images = [c for c in crops if c is not None]
        if len(images) < 2:
            return None
        clip_img, dino_img = self.embed(images)
        clip_mean, clip_min, clip_pairs = _mean_pairwise(clip_img)
        dino_mean, dino_min, dino_pairs = _mean_pairwise(dino_img)
        out = {
            "n_images": len(images),
            "clip_i_self_consistency_mean": round(clip_mean, 4),
            "clip_i_self_consistency_worst_pair": round(clip_min, 4),
            "dino_self_consistency_mean": round(dino_mean, 4),
            "dino_self_consistency_worst_pair": round(dino_min, 4),
            "per_pair_dino": [round(s, 4) for s in dino_pairs],
        }
        if face_crop:
            out["face_crop"] = True
            out["images_without_face"] = missing
        if scene_emb is not None and len(images) == len(SCENES):
            img_n = _l2(clip_img)
            clip_t = [float(img_n[i] @ scene_emb[i]) for i in range(len(images))]
            out["clip_t_scene_fidelity_mean"] = round(float(np.mean(clip_t)), 4)
            out["per_image_clip_t"] = [round(s, 4) for s in clip_t]
        return out


def evaluate(out_dir, face_crop=False):
    scorer = Scorer()
    scene_emb = scorer.text(SCENES)
    results = {}
    for arm_name in ARMS:
        arm_dir = os.path.join(out_dir, arm_name)
        if os.path.isdir(arm_dir):
            r = scorer.score_folder(arm_dir, scene_emb, face_crop)
            if r:
                r["config"] = ARMS[arm_name]
                results[arm_name] = r
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def contact_sheet(out_dir, thumb=384):
    for arm_name in ARMS:
        arm_dir = os.path.join(out_dir, arm_name)
        if not os.path.isdir(arm_dir):
            continue
        images = [im.resize((thumb, thumb)) for im in _load_images(arm_dir)]
        cols = 3
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), "white")
        for idx, im in enumerate(images):
            sheet.paste(im, ((idx % cols) * thumb, (idx // cols) * thumb))
        sheet.save(os.path.join(out_dir, f"grid_{arm_name}.png"))


def print_table(results):
    header = (
        f"{'arm':<14}{'DINO self':>11}{'DINO worst':>12}"
        f"{'CLIP-I self':>13}{'CLIP-T scene':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for arm, r in results.items():
        print(
            f"{arm:<14}{r['dino_self_consistency_mean']:>11.4f}"
            f"{r['dino_self_consistency_worst_pair']:>12.4f}"
            f"{r['clip_i_self_consistency_mean']:>13.4f}"
            f"{r.get('clip_t_scene_fidelity_mean', float('nan')):>14.4f}"
            + (f"   faces {r['n_images']}/{r['n_images'] + len(r['images_without_face'])}"
               if r.get("face_crop") else "")
        )
    print()


def compare(path_a, path_b):
    a = json.load(open(path_a))["arms"]
    b = json.load(open(path_b))["arms"]
    keys = [
        "dino_self_consistency_mean",
        "clip_i_self_consistency_mean",
        "clip_t_scene_fidelity_mean",
    ]
    print(f"\n{os.path.basename(path_a)}  ->  {os.path.basename(path_b)}\n")
    for arm in a:
        if arm not in b:
            continue
        print(f"  {arm}")
        for k in keys:
            if k in a[arm] and k in b[arm]:
                d = b[arm][k] - a[arm][k]
                print(f"    {k:<34}{a[arm][k]:>8.4f} -> {b[arm][k]:>8.4f}  ({d:+.4f})")
        print()
    print("  Gaps under ~0.05 at this sample size are not conclusive.\n")


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/baseline")
    ap.add_argument("--character", default=CHARACTER,
                    help='For a LoRA run, pass the trained token, e.g. "sks woman".')
    ap.add_argument("--lora_dir", default=None)
    ap.add_argument("--lora_scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--eval_dir", default=None,
                    help="Score one flat folder for self-consistency and exit.")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    ap.add_argument("--face_crop", action="store_true",
                    help="Score face crops instead of whole frames. Writes eval_face.json.")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.eval_dir:
        r = Scorer().score_folder(args.eval_dir, face_crop=args.face_crop)
        print(json.dumps(r, indent=2))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if not args.eval_only:
        generate(args.out_dir, args.character, args.steps, args.seed,
                 args.size, args.lora_dir, args.lora_scale)

    results = evaluate(args.out_dir, face_crop=args.face_crop)
    if not args.face_crop:
        contact_sheet(args.out_dir)

    payload = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "lora_dir": args.lora_dir,
        "lora_scale": args.lora_scale if args.lora_dir else None,
        "character": args.character,
        "scenes": SCENES,
        "steps": args.steps,
        "base_seed": args.seed,
        "resolution": args.size,
        "face_crop": args.face_crop,
        "arms": results,
    }
    out_json = os.path.join(args.out_dir, "eval_face.json" if args.face_crop else "eval.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print_table(results)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
