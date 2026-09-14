"""Generate a scene grid from a fine-tuned SDXL + LoRA character model.

Loads the SDXL base pipeline, applies the trained LoRA weights, and renders the
same character across a list of prompts with a fixed seed so results are
reproducible and comparable across runs.
"""
import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image


def load_prompts(prompts_file: str | None) -> list[str]:
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file) as f:
            return [line.strip() for line in f if line.strip()]
    # Fallback so the script runs even without a prompts file.
    return [
        "a portrait of sks person, studio lighting, high detail",
        "sks person as an astronaut floating in space, cinematic",
        "sks person sitting in a cozy coffee shop, warm light",
        "sks person hiking a mountain trail at sunrise",
        "sks person in a neon-lit cyberpunk city at night",
        "sks person wearing medieval knight's armor",
    ]


def make_grid(images: list[Image.Image], cols: int = 3) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h), "white")
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * w, (i // cols) * h))
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", required=True, help="Dir with trained LoRA weights")
    p.add_argument("--base_model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--prompts_file", default="scripts/prompts.txt")
    p.add_argument("--out_dir", default="outputs/samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--grid_cols", type=int, default=3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[warn] No GPU detected. SDXL inference on CPU is extremely slow.")
    dtype = torch.float16 if device == "cuda" else torch.float32

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, vae=vae, torch_dtype=dtype
    ).to(device)
    pipe.load_lora_weights(args.lora_dir)
    pipe.set_progress_bar_config(disable=True)

    prompts = load_prompts(args.prompts_file)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    images = []
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        img = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        out_path = os.path.join(args.out_dir, f"gen_{i:02d}.png")
        img.save(out_path)
        images.append(img)

    grid = make_grid(images, cols=args.grid_cols)
    grid_path = os.path.join(args.out_dir, "grid.png")
    grid.save(grid_path)
    print(f"Saved {len(images)} images + grid -> {grid_path}")


if __name__ == "__main__":
    main()
