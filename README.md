# Character Consistency with SDXL LoRA

How well does Stable Diffusion XL keep one character's identity across different scenes, and how much does a DreamBooth LoRA fine-tune add on top of prompting?

This repo measures that with a fixed character, a fixed set of scenes, fixed seeds, and one shared set of metric definitions. Full tables are in [RESULTS.md](RESULTS.md).

## Headline result

Adding the LoRA to the same text description makes faces markedly more consistent across scenes. Identity is measured with **SFace**, a face-recognition embedding, on detected and aligned faces (higher is more similar):

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only (varied seed, CFG 7) | 0.525 | 0.613 | **0.751** |
| fixed_seed (one seed, CFG 7) | 0.545 | 0.679 | **0.809** |
| high_cfg (one seed, CFG 12) | 0.648 | 0.675 | **0.798** |

- **Text + LoRA vs text only** isolates the adapter, since the prompts are otherwise identical: identity rises by 0.15 to 0.26 in every arm, and the *worst pair* rises by 0.12 to 0.32. The adapter removes failures, not just raises the average.
- **The LoRA token alone also beats the text description** in every arm, so the adapter carries identity on its own.
- **Scene fidelity cost is small.** Whole-frame CLIP-T fell by 0.025 in `prompt_only` and was flat in the other two arms.

Full tables, including the DINOv2 and CLIP metrics on whole frames and face crops, are in [RESULTS.md](RESULTS.md).

## What we learned

**Metric specificity decided the answer three times, and the more identity-specific metric was right every time.**

1. **Whole-frame scoring said the fine-tune made things worse.** Scored on whole images, the token-only LoRA fell from 0.24 to 0.16 on DINOv2. The prompts differed: the baseline described the face in detail, while `sks woman` left the scene words more weight, so compositions spread into full-body and wide shots. Whole-frame similarity dropped because the frames differed, not the faces.
2. **Face-cropped DINOv2 fixed that, and revised a second conclusion.** On whole frames a fixed seed looked unimportant (0.24 to 0.29); on face crops it was large (0.50 to 0.66). But face-cropped DINOv2 still said the token alone underperformed the description.
3. **Face-recognition scoring revised that too.** DINOv2 responds to framing, hair and expression even after cropping. SFace aligns the face first and is trained to ignore everything but identity, and by that measure the token alone *does* beat the description. The general lesson: a metric answers the question it was trained on, not the question you are asking.

**CLIP cannot tell these faces apart at all.** On whole frames CLIP-I was about 0.80 while DINOv2 was 0.24 on the same images. On face crops CLIP-I sits between 0.80 and 0.87 in every condition, while DINOv2 ranges 0.50 to 0.71 and SFace 0.52 to 0.81. CLIP matches images to captions, so any auburn-haired, freckled woman scores alike.

**The adapter is more consistent than its own training data.** The 12-image training set scores 0.722 on SFace, while text + LoRA with a fixed seed reaches 0.809. The adapter learns a central identity averaged over imperfect examples rather than being capped by the worst of them.

**Text conditioning produces an attribute bundle, not a person.** Every baseline image kept red curly hair, freckles and green eyes, while face shape, apparent age and hair length drifted. The scar above the left eyebrow never appeared, in the baseline or in the training hero, so the LoRA had no way to learn it.

**Img2img from one frontal reference cannot produce pose diversity.** The training-set prompts asked for profile and three-quarter views and different lighting; none appeared. Raising strength from 0.5 to 0.65 lowered the set's DINOv2 self-consistency only from 0.971 to 0.950. Img2img preserves spatial layout, so a single frontal reference yields near-duplicates.

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
| SFace identity | SFace, on YuNet-detected and aligned faces | Identity similarity between generations |
| DINO self-consistency | DINOv2-base CLS token | Instance-level similarity between generations |
| CLIP-I self-consistency | CLIP ViT-L/14 image embeddings | Semantic similarity between generations |
| CLIP-T scene fidelity | CLIP ViT-L/14 image vs scene text | Whether the scene was followed (whole frames only) |

CLIP and DINOv2 metrics are computed on whole frames and on face crops (OpenCV's Haar detector, largest face, 30% margin). SFace scores faces detected by YuNet and aligned to a canonical five-point layout, which removes in-plane rotation and scale. A face was found in all 54 generated images and all 12 training images.

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

**6. Score identity with a face-recognition embedding:**

```bash
for d in baseline lora lora_desc; do
  python eval/face_identity.py --out_dir results/$d
done
python eval/face_identity.py --compare results/baseline/eval_sface.json results/lora_desc/eval_sface.json
python eval/face_identity.py --dir data/training_set_s065
```

The YuNet and SFace models download on first use into `models/`, which is gitignored.

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
- **DINO on a face crop is not a face-recognition metric.** It responds to pose, expression, lighting and hair as well as identity, which is why the SFace numbers are the headline and the DINOv2 ones are context.
- **SFace was trained on photographs of real people,** so synthetic faces are out of distribution and absolute values should be read with care. OpenCV's 0.363 same-identity threshold is uninformative here: every pair in every condition passes it, including the baseline, because all these faces share hair, freckles and eye colour.
- **Low-diversity synthetic training data.** All twelve training images are frontal head-and-shoulders portraits with the same clothing and background.
- **One training run, one seed schedule, one base model.** No claims are made about variance across runs or other backbones.

## Next steps

- Compare the step-400 and step-800 checkpoints and sweep LoRA scale to trace the identity versus editability tradeoff.
- Build a more diverse training set with multi-view generation or pose conditioning, and retest the token-only condition.
- Add prior preservation and measure class drift.
