#!/usr/bin/env bash
# SDXL DreamBooth + LoRA on a 24GB card with bf16 (L4, A10, A100).
# For a 16GB T4, use train_lora_t4.sh instead.
#
# The two scripts share every modeling choice (steps, LR, schedule, token)
# and differ only in flags forced by hardware, so results are comparable.
#
#   resolution 1024         SDXL's native resolution.
#   mixed_precision bf16    Same exponent range as fp32, so no overflow to NaN.
#                           The fp16-fix VAE is kept anyway as belt and braces.
#   gradient_checkpointing  Recomputes activations in the backward pass.
#                           Needed at 1024 even on 24GB.
#   rank 16                 LoRA capacity. Higher locks identity harder and
#                           risks memorizing pose, background and clothing.
#   800 steps, batch 1      ~67 epochs over 12 images. Checkpoint at 400 gives
#                           an undertrained comparison point.
#   no 8-bit Adam           For LoRA the trainable parameters are small, so
#                           optimizer state is a few hundred MB at most.
#                           8-bit Adam matters for full fine-tuning, not here.
#   no xformers             PyTorch 2 SDPA is the default and equivalent.
#   no prior preservation   Skipped for time. The known cost is class drift,
#                           where every "woman" starts to look like sks.
#
# Usage:
#   bash scripts/train_lora.sh
#   MAX_STEPS=10 bash scripts/train_lora.sh      # smoke test

set -euo pipefail

INSTANCE_DIR="${INSTANCE_DIR:-data/my_character}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of sks woman}"
MAX_STEPS="${MAX_STEPS:-800}"
RANK="${RANK:-16}"
LR="${LR:-1e-4}"
SEED="${SEED:-1234}"

# Fetch the training script tagged for the INSTALLED diffusers version.
# Scripts on main usually require an unreleased dev version and refuse to run.
SCRIPT="train_dreambooth_lora_sdxl.py"
if [ ! -f "$SCRIPT" ]; then
  V=$(python -c "import diffusers; print(diffusers.__version__)")
  echo "Fetching $SCRIPT for diffusers v$V..."
  curl -fsSL -o "$SCRIPT" \
    "https://raw.githubusercontent.com/huggingface/diffusers/v${V}/examples/dreambooth/${SCRIPT}"
fi

accelerate launch "$SCRIPT" \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate="$LR" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank="$RANK" \
  --max_train_steps="$MAX_STEPS" \
  --checkpointing_steps=400 \
  --seed="$SEED"

echo
echo "Adapter: $OUTPUT_DIR"
echo "Step-400 checkpoint: $OUTPUT_DIR/checkpoint-400"

# Prior preservation, if you want to test class drift later:
#   --with_prior_preservation --prior_loss_weight=1.0 \
#   --class_prompt="a photo of a woman" --class_data_dir="data/class_woman" \
#   --num_class_images=200
