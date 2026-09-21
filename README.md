# Character Consistency with SDXL LoRA

How well does Stable Diffusion XL keep one character's identity across different scenes, and how much does a DreamBooth LoRA fine-tune add on top of prompting?

This repo measures that with a fixed character, a fixed set of scenes, fixed seeds, and one shared set of metric definitions. Full tables are in [RESULTS.md](RESULTS.md).

## Headline result

Adding the LoRA to the same text description makes faces more consistent across scenes. Face-cropped DINOv2 self-consistency, where higher means the six faces in an arm look more alike:

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only (varied seed, CFG 7) | 0.50 | 0.55 | **0.62** |
| fixed_seed (one seed, CFG 7) | 0.66 | 0.55 | **0.70** |
| high_cfg (one seed, CFG 12) | 0.62 | 0.57 | **0.71** |

- **Text + LoRA vs text only** isolates the adapter, since the prompts are otherwise identical. Gains of 0.11 and 0.09 in two arms are clear; 0.04 in the fixed-seed arm is within noise at this sample size. In `prompt_only`, the worst pair rises from 0.22 to 0.41, so the adapter mostly removes outlier faces.
- **The LoRA token alone does not beat the text description.** Trained on twelve near-identical portraits, `sks woman` complements the description rather than replacing it.
- **Scene fidelity cost is small.** Whole-frame CLIP-T fell by 0.025 in `prompt_only` and was flat in the other two arms.

## What we learned

**Whole-frame metrics misled twice.** Scored on whole images, the token-only LoRA looked worse than the baseline (DINO 0.24 down to 0.16). The prompts differed: the baseline spelled out the face in detail, while `sks woman` left the scene words more weight, so compositions spread out into full-body and wide shots. Whole-frame similarity dropped because the frames differed, not because the faces did. Whole-frame scoring also hid how much a fixed seed helps (below). Scoring face crops fixed both.

**A fixed seed is a strong face-consistency lever on its own.** On whole frames, fixing the seed moved DINO only from 0.24 to 0.29. On face crops it moves from 0.50 to 0.66. Some of that is likely shared head angle and lighting from the same starting noise rather than identity, since DINO sees pose and expression too.

**CLIP cannot tell these faces apart.** On whole frames, CLIP-I was about 0.80 while DINOv2 was 0.24 on the same images. On face crops, CLIP-I sits between 0.80 and 0.87 across every condition while DINO spreads from 0.50 to 0.71. CLIP matches images to semantic descriptions, so any auburn-haired, freckled woman scores alike. DINOv2 is instance-discriminative.

**Text conditioning produces an attribute bundle, not a person.** Every baseline image kept red curly hair, freckles and green eyes. Face shape, apparent age and hair length drifted. The scar above the left eyebrow never appeared, in the baseline or in the training hero, so the LoRA had no way to learn it.

**Img2img from one frontal reference cannot produce pose diversity.** The training-set prompts asked for profile and three-quarter views, different lighting and darker backgrounds; none appeared. Raising strength from 0.5 to 0.65 lowered the set's DINO self-consistency only from 0.971 to 0.950. Img2img preserves spatial layout, so a single frontal reference yields near-duplicates. This is the most likely reason the token alone did not transfer identity to new scenes.

## Method

**Character:** a 28-year-old woman with short curly auburn hair, light freckles across the nose, green eyes, and a small scar above the left eyebrow.

**Scenes:** coffee shop, rainy street at night, mountain trail, space station in a spacesuit, stone courtyard in medieval armor, summer picnic.

**Arms:** `prompt_only` (varied seed, CFG 7), `fixed_seed` (one seed, CFG 7), `high_cfg` (one seed, CFG 12). SDXL base, 30 steps, 1024 x 1024.

**Conditions:**

| Condition | Prompt subject | Adapter |
| --- | --- | --- |
| Text only | Full character description | None |
| LoRA token only | `sks woman` | LoRA |
| Text + LoRA | `sks woman` followed by the full description | LoRA |

**Metrics:**

| Metric | Model | What it measures |
| --- | --- | --- |
| DINO self-consistency | DINOv2-base CLS token | Instance-level similarity between generations |
| CLIP-I self-consistency | CLIP ViT-L/14 image embeddings | Semantic similarity between generations |
| CLIP-T scene fidelity | CLIP ViT-L/14 image vs scene text | Whether the scene was followed (whole frames only) |

Each metric is computed on whole frames and on face crops. Face crops use OpenCV's Haar detector on the largest face with a 30% margin; all 54 generated images had a detected face.

**Training set:** one txt2img hero portrait plus eleven img2img variations at strength 0.65, all synthetic.

**Training:** DreamBooth LoRA on the SDXL UNet with the official diffusers script. Rank 16, learning rate 1e-4 constant, 800 steps at batch size 1, 1024px, bf16, text encoders frozen, no prior preservation. The instance prompt names the shared sweater and backdrop so they bind to those words rather than the token. About 20 minutes on an L4.

## Repository layout

```
eval/consistency_experiment.py   Generation and scoring harness: baseline arms, LoRA arms,
                                 whole-frame and face-cropped scoring, single-folder
                                 diagnostics, and before/after comparison
scripts/make_character_set.py    Builds the synthetic training set (hero + img2img variations)
scripts/train_lora.sh            Training config for 24GB cards with bf16 (L4, A10, A100)
scripts/train_lora_t4.sh         Training config for 16GB T4 cards
notebook/run_experiments_t4.ipynb  Colab runner for the full pipeline (runs on T4 or L4)
data/                            Training images (gitignored), see data/README.md
results/                         Metrics, JSON and contact sheets (tracked)
RESULTS.md                       Full results tables and notes
```

## Running it

The pipeline runs on Google Colab. An L4 (24GB) is recommended; a T4 (16GB) works with the T4 training script.

**Setup:**

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/Colin-J-Emmanuel/character-consistency-lora.git
%cd character-consistency-lora
!pip install -q diffusers transformers accelerate peft safetensors bitsandbytes ftfy datasets
!pip install -q "protobuf>=5.29.1,<6"
!pip uninstall -y -q torchao
!pip install -q "opencv-python-headless<5"
!mkdir -p /content/drive/MyDrive/ccl && ln -sfn /content/drive/MyDrive/ccl outputs
```

Restart the Colab session after the installs, then re-run the Drive mount and `%cd` lines. `outputs/` is linked to Google Drive so trained adapters survive a disconnect.

**1. Baseline (text only):**

```bash
python eval/consistency_experiment.py --out_dir results/baseline
```

**2. Training set, then its diagnostics:**

```bash
python scripts/make_character_set.py --out_dir data/my_character --strength 0.65
python eval/consistency_experiment.py --eval_dir data/my_character
```

Look at the images before training. Remove any where the face is visibly a different person.

**3. Train.** Smoke test first, then the full run:

```bash
MAX_STEPS=10 bash scripts/train_lora.sh
rm -rf outputs/lora
INSTANCE_PROMPT="a photo of sks woman, sage green sweater, plain grey backdrop" bash scripts/train_lora.sh
```

On a T4, use `scripts/train_lora_t4.sh` instead. The two scripts share steps, learning rate, schedule and token, and differ only in hardware-driven flags.

| Card | Script | Resolution | Precision | Memory measures |
| --- | --- | --- | --- | --- |
| L4, A10, A100 (24GB+) | `train_lora.sh` | 1024 | bf16 | Gradient checkpointing |
| T4 (16GB) | `train_lora_t4.sh` | 768 | fp16 with fixed VAE | Gradient checkpointing, 8-bit Adam |

**4. Evaluate both LoRA conditions:**

```bash
python eval/consistency_experiment.py --out_dir results/lora --lora_dir outputs/lora --character "sks woman"
python eval/consistency_experiment.py --out_dir results/lora_desc --lora_dir outputs/lora \
    --character "sks woman, a 28-year-old woman with short curly auburn hair, light freckles across the nose, green eyes, and a small scar above the left eyebrow"
```

**5. Score face crops and compare:**

```bash
for d in baseline lora lora_desc; do
  python eval/consistency_experiment.py --out_dir results/$d --eval_only --face_crop
done
python eval/consistency_experiment.py --compare results/baseline/eval_face.json results/lora_desc/eval_face.json
```

`--lora_scale` sets adapter strength, and `--lora_dir outputs/lora/checkpoint-400` evaluates the intermediate checkpoint.

## Engineering notes

These came up while running on current Colab images and are handled in the code or setup.

- **SDXL's stock VAE produces NaNs in fp16.** All pipelines load `madebyollin/sdxl-vae-fp16-fix`. On a T4, which has no bf16, this is what makes fp16 work at all.
- **diffusers 0.39 moved VAE slicing** from the pipeline to `pipe.vae.enable_slicing()`.
- **transformers 5 changed CLIP's `get_text_features` and `get_image_features`** to return a model output instead of a tensor. The harness calls the text and vision submodules and applies the projection explicitly. Using `pooler_output` directly would compare 768-dim text states against 1024-dim vision states outside the shared CLIP space.
- **Pipeline placement depends on available memory.** Cards with 20GB or more keep the whole pipeline on the GPU; smaller cards use CPU offload.
- **The training script is fetched for the installed diffusers version.** Scripts on the diffusers `main` branch usually require an unreleased dev version.
- **PEFT refuses to run with Colab's preinstalled torchao 0.10.** The project does not use torchao, so setup uninstalls it.
- **The `diffusers[training]` extra pins protobuf below 4**, which conflicts with Colab's Google libraries. Setup installs the needed packages individually and pins protobuf to a compatible range.
- **OpenCV 5 removed the Haar cascade face detector** from the main package. Face-cropped scoring needs the 4.x line, pinned in setup.
- **LoRA scale is set through `set_adapters`**, which works under the PEFT backend.
- **Diffusion training loss is not a quality signal.** Per-step loss mostly reflects which noise level was sampled, so evaluation uses generated images on held-out scenes instead.

## Limitations

- **Small samples.** Six images per arm gives 15 pairs. Treat gaps below about 0.05 as inconclusive.
- **DINO on a face crop is not a face-recognition metric.** It responds to pose, expression, lighting and hair as well as identity. A face-recognition embedding such as ArcFace would measure identity more directly.
- **Simple face detection.** The Haar detector is crude next to a learned detector. It found every face here, but distant or occluded faces could be missed or cropped poorly.
- **Low-diversity synthetic training data.** All twelve training images are frontal head-and-shoulders portraits with the same clothing and background.
- **One training run, one seed schedule, one base model.** No claims are made about variance across runs or other backbones.

## Next steps

- Score identity with a face-recognition embedding (ArcFace or OpenCV's SFace).
- Compare the step-400 and step-800 checkpoints and sweep LoRA scale to trace the identity versus editability tradeoff.
- Build a more diverse training set with multi-view generation or pose conditioning, and retest the token-only condition.
- Add prior preservation and measure class drift.
