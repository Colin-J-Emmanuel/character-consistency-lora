# Character-Consistent Generation with SDXL + LoRA

Fine-tune Stable Diffusion XL with LoRA (via DreamBooth) to generate a **single
consistent character** across arbitrary scenes and prompts, then measure that
consistency with a rigorous, identity-aware eval harness.

This is the core problem behind AI character platforms: the same character has to
look like *itself* whether it's in a coffee shop, in space, or in armor.

## Why this approach

- **DreamBooth + LoRA on SDXL.** DreamBooth teaches the base model a new subject
  via a rare identifier token (e.g. `sks person`). LoRA keeps it cheap and portable:
  we only train small low-rank adapter matrices (a few MB) instead of the full UNet.
- **Prior-preservation (optional)** keeps the model from collapsing / forgetting the
  general "person" class while it learns the specific character.
- **Identity-aware eval.** Prompt-fidelity alone (CLIP-T) doesn't tell you if the
  character is *consistent*. We add CLIP-I and DINO image-image similarity for
  identity fidelity, plus gen-gen self-consistency across scenes.

## Metrics (what each one answers)

| Metric | Question it answers | Direction | Typical range |
|---|---|---|---|
| **CLIP-T** | Does the image match the prompt? | higher = better | ~0.25–0.35 |
| **CLIP-I** | Does the generated character match the reference photos? | higher = better | ~0.6–0.85 |
| **DINO**   | Same, but instance-sensitive (finer identity detail) | higher = better | ~0.4–0.8 |
| **Self-consistency** | Do the generations look like the *same* character across scenes? | higher = better | ~0.6–0.85 |

CLIP-T values look "low" because text–image cosine similarity never approaches 1;
what matters is relative movement (base model vs. fine-tuned). DINO is stricter than
CLIP-I because it keys on instance-level detail rather than semantic category — be
ready to explain that distinction.

## Repo layout

```
character-consistency-lora/
├── data/my_character/        # 8–15 photos of ONE character (varied angle/lighting)
├── scripts/
│   ├── train_lora.sh         # wrapper around diffusers DreamBooth-LoRA-SDXL
│   └── generate.py           # load base + LoRA, generate a scene grid
├── eval/
│   ├── metrics.py            # CLIP-T / CLIP-I / DINO / self-consistency
│   └── evaluate.py           # CLI: runs the full eval, writes eval_results.json
├── outputs/samples/          # generated grids
└── requirements.txt
```

## Quickstart (needs a GPU — T4 16GB minimum, A10/L4/A100 ideal)

```bash
pip install -r requirements.txt

# 1. Put 8–15 images of your character in data/my_character/
#    Varied poses/lighting, consistent identity, cropped to the subject.

# 2. Train the LoRA (~20–45 min on an A10; longer on a T4)
bash scripts/train_lora.sh

# 3. Generate a scene grid from the fine-tuned model
python scripts/generate.py --lora_dir outputs/lora --out_dir outputs/samples

# 4. Evaluate consistency
python eval/evaluate.py \
    --ref_dir data/my_character \
    --gen_dir outputs/samples \
    --prompts_file scripts/prompts.txt \
    --out outputs/eval_results.json
```

## Interview talking points (be able to defend each)

- Why LoRA over full fine-tuning; what rank trades off (capacity vs. overfitting).
- What the rare token does and why prior-preservation matters.
- SDXL's fp16 VAE NaN issue → why the `sdxl-vae-fp16-fix` VAE.
- Why CLIP-T alone is insufficient for a consistency task, and why DINO is stricter
  than CLIP-I.
- Failure modes you saw: overfitting (same pose every time), identity drift at high
  guidance, prompt terms the LoRA "swallows."

## Honest scope

This is a fine-tuning + evaluation project on a pretrained SDXL backbone — not a
diffusion model trained from scratch. (See the companion `ddpm-from-scratch` repo
for the fundamentals.) Results and limitations are reported as measured.
