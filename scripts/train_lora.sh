#!/usr/bin/env bash
# Train a character LoRA on SDXL using the official diffusers DreamBooth script.
# Using the maintained script (not a hand-rolled loop) is the standard, defensible
# choice — but you should understand every flag below, because that's what an
# interviewer will probe.
set -euo pipefail

# --- Get the official training script (one-time) ---------------------------
SCRIPT="train_dreambooth_lora_sdxl.py"
if [ ! -f "$SCRIPT" ]; then
  echo "Fetching $SCRIPT from diffusers..."
  curl -L -o "$SCRIPT" \
    https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/${SCRIPT}
fi

# --- Config ----------------------------------------------------------------
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's default VAE produces NaNs in fp16; this fixed VAE is the standard workaround.
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export INSTANCE_DIR="data/my_character"
export OUTPUT_DIR="outputs/lora"

# 'sks' is a rare token with little prior meaning, so the model binds it to YOUR
# character instead of overwriting a common word. Keep the class word ("person",
# or "man"/"woman"/"character") accurate — it anchors the prior.
INSTANCE_PROMPT="a photo of sks person"
CLASS_PROMPT="a photo of a person"          # only used if prior-preservation is on
VALIDATION_PROMPT="a photo of sks person hiking a mountain at sunrise"

accelerate launch "$SCRIPT" \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --pretrained_vae_model_name_or_path="$VAE_NAME" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=16 \
  --max_train_steps=1000 \
  --checkpointing_steps=250 \
  --validation_epochs=25 \
  --seed=42 \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention

# ---------------------------------------------------------------------------
# HYPERPARAMETER NOTES (know these cold)
#   rank=16              LoRA capacity. 8=lighter/less overfit, 32=more capacity.
#   max_train_steps=1000 Single subject: 800–1500 is the usual band. Too many
#                        steps = overfitting (character always in the same pose).
#   learning_rate=1e-4   Standard for LoRA; full fine-tunes use far smaller LRs.
#   resolution=1024      SDXL native. Drop to 768 if you're VRAM-limited (T4).
#   grad_checkpointing + 8bit_adam + xformers  = the trio that fits SDXL LoRA
#                        onto ~16GB. Trades compute/precision for memory.
#
# PRIOR PRESERVATION (optional, reduces overfitting/forgetting):
#   add these flags and generate ~100–200 class images first:
#     --with_prior_preservation --prior_loss_weight=1.0 \
#     --class_prompt="$CLASS_PROMPT" --class_data_dir="data/class_person" \
#     --num_class_images=200
# ---------------------------------------------------------------------------
