#!/usr/bin/env python
"""
Build a DreamBooth training set for ONE synthetic character.

Why not just prompt twelve times with the same description? Because that is
exactly what Experiment 1 measures, and it does not hold identity. Instead:

    1. Generate one hero portrait (txt2img, fixed seed).
    2. Produce variations from it with img2img at moderate strength, so the
       face structure is inherited from the hero while pose, lighting,
       expression and background vary.

Strength is the dial. Low strength gives near-duplicates, which teaches the
LoRA one pose and nothing about the person. High strength gives variety but
drifts identity, which poisons the set. 0.5 is a reasonable starting point.

The set is synthetic on purpose: no likeness rights issues, and it is
reproducible from a seed. The honest cost is that the training set is only as
coherent as img2img makes it, which caps achievable identity fidelity. Measure
that ceiling before training:

    python eval/consistency_experiment.py --eval_dir data/my_character

Usage:
    python scripts/make_character_set.py --out_dir data/my_character
"""

import argparse
import os

import torch

CHARACTER = (
    "a 28-year-old woman with short curly auburn hair, light freckles across "
    "the nose, green eyes, and a small scar above the left eyebrow"
)

HERO = (
    f"a photo of {CHARACTER}, head and shoulders portrait, neutral expression, "
    "soft studio lighting, plain grey backdrop, sharp focus, 85mm"
)

# Vary pose, lighting, expression, framing and background. Keep wardrobe and
# setting plain so the LoRA binds to the person, not to a jacket or a room.
VARIATIONS = [
    "three-quarter view portrait, soft window light, plain wall background",
    "profile view portrait, warm side lighting, plain background",
    "smiling warmly, natural daylight, plain background",
    "serious expression, overcast outdoor light, plain background",
    "looking slightly away from camera, golden hour light, plain background",
    "close-up face, even studio lighting, plain background",
    "head tilted, soft diffused light, plain background",
    "laughing, bright daylight, plain background",
    "chin slightly down, dramatic rim lighting, dark plain background",
    "upper body, standing, flat frontal lighting, plain background",
    "slight upward angle, cool indoor light, plain background",
]

NEGATIVE = (
    "blurry, lowres, deformed, extra limbs, watermark, text, cartoon, "
    "illustration, multiple people"
)

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
FIXED_VAE = "madebyollin/sdxl-vae-fp16-fix"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/my_character")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--n", type=int, default=12, help="total images including hero")
    args = ap.parse_args()

    from diffusers import (
        AutoencoderKL,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
    )

    if not torch.cuda.is_available():
        raise SystemExit("Needs a GPU. Colab: Runtime > Change runtime type > T4 GPU.")

    os.makedirs(args.out_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(FIXED_VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, vae=vae, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

    hero = pipe(
        prompt=HERO,
        negative_prompt=NEGATIVE,
        num_inference_steps=args.steps,
        guidance_scale=7.0,
        height=args.size,
        width=args.size,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    ).images[0]
    hero.save(os.path.join(args.out_dir, "00_hero.png"))
    print("hero saved")

    # Reuse the loaded components, no second 7GB download or second copy in RAM.
    i2i = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    i2i.set_progress_bar_config(disable=True)
    i2i.enable_model_cpu_offload()

    for i, variation in enumerate(VARIATIONS[: args.n - 1]):
        img = i2i(
            prompt=f"a photo of {CHARACTER}, {variation}, sharp focus, 85mm",
            negative_prompt=NEGATIVE,
            image=hero,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=7.0,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + i + 1),
        ).images[0]
        img.save(os.path.join(args.out_dir, f"{i + 1:02d}.png"))
        print(f"variation {i + 1}: {variation}")

    print(f"\n{args.n} images in {args.out_dir}")
    print("Now measure the ceiling your training data imposes:")
    print(f"  python eval/consistency_experiment.py --eval_dir {args.out_dir}")
    print("Then DELETE any image where the face is visibly a different person.")


if __name__ == "__main__":
    main()
