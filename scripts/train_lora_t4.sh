#!/usr/bin/env bash
# SDXL DreamBooth + LoRA on a free Colab T4 (16GB, Turing, no bf16).
#
# Every flag below is either a memory concession or a deliberate choice.
# Be able to defend each one:
#
#   resolution 768        1024 OOMs on 16GB with this script. 768 fits.
#   gradient_checkpointing  recomputes activations in the backward pass.
#                         Trades roughly 30% speed for a large memory saving.
#   cache_latents         encodes the training images once, then evicts the
#                         VAE from GPU. Frees ~1GB for the whole run.
#   use_8bit_adam         optimizer states in 8-bit instead of fp32. Adam keeps
#                         two moments per parameter, so this is real savings.
#   mixed_precision fp16  T4 is Turing, bf16 is unavailable. fp16 with SDXL is
#                         NaN-prone through the stock VAE, which is exactly why
#                         pretrained_vae_model_name_or_path points at the fix.
#   rank 8                LoRA capacity. Higher locks identity harder and
#                         overfits pose and background. Lower underfits the
#                         face. 8 is the middle of the usual 4 to 16 band.
#   no train_text_encoder Saves memory, and text-encoder training on a small
#                         set is a known overfitting risk.
#   no prior preservation Skipped on this run for time and memory. The cost is
#                         no regularization against class drift, so watch for
#                         the model turning every "woman" into this character.
#                         Named as a limitation rather than hidden.
#
# Usage:  bash scripts/train_lora_t4.sh

set -euo pipefail

INSTANCE_DIR="${INSTANCE_DIR:-data/my_character}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks woman}"
MAX_STEPS="${MAX_STEPS:-800}"
RANK="${RANK:-8}"
LR="${LR:-1e-4}"
SEED="${SEED:-1234}"

if [ ! -f "train_dreambooth_lora_sdxl.py" ]; then
  echo "Fetching the diffusers training script..."
  wget -q https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py
fi

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="${INSTANCE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --mixed_precision="fp16" \
  --resolution=768 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --cache_latents \
  --use_8bit_adam \
  --learning_rate="${LR}" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank="${RANK}" \
  --max_train_steps="${MAX_STEPS}" \
  --checkpointing_steps=400 \
  --seed="${SEED}"

echo
echo "Done. Adapter in ${OUTPUT_DIR}"
echo "Intermediate checkpoint at step 400 is in ${OUTPUT_DIR}/checkpoint-400"
echo "Comparing 400 vs 800 shows the identity-versus-editability tradeoff."
