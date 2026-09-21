# Character Consistency with SDXL LoRA

How well does Stable Diffusion XL keep one character's identity across different scenes, and how much does a DreamBooth LoRA fine-tune improve on prompting alone?

This repo measures that with a fixed character, a fixed set of scenes, fixed seeds, and one shared set of metric definitions, so the baseline and fine-tuned runs are directly comparable.

**Status:** the prompting baseline and the training-set diagnostics are complete. The LoRA fine-tune is in progress; its results will be added below and in [RESULTS.md](RESULTS.md).

## Results so far

### Experiment 1: prompting alone

One detailed character description, six scenes, SDXL base, three arms that vary only seed strategy and guidance scale. Identity is mean pairwise similarity across each arm's six images. Scene fidelity is CLIP similarity to the scene text alone, with the character description excluded, so identity and prompt-following stay separable.

| Arm | Seed | CFG | DINO self-consistency | DINO worst pair | CLIP-I self-consistency | CLIP-T scene |
| --- | --- | --- | --- | --- | --- | --- |
| prompt_only | varied | 7.0 | 0.239 | 0.060 | 0.800 | 0.233 |
| fixed_seed | fixed | 7.0 | 0.291 | 0.130 | 0.822 | 0.209 |
| high_cfg | fixed | 12.0 | 0.302 | 0.114 | 0.795 | 0.213 |

Grids: `results/baseline/grid_*.png`. Raw numbers: `results/baseline/eval.json`.

### Training set diagnostics

The training set is one txt2img hero portrait plus eleven img2img variations of it. Before training, its self-consistency was measured with the same metrics.

| img2img strength | DINO self-consistency | DINO worst pair | CLIP-I self-consistency |
| --- | --- | --- | --- |
| 0.50 | 0.971 | 0.942 | 0.975 |
| 0.65 | 0.950 | 0.884 | 0.965 |

The 0.65 set is the one used for training. Grid: `results/training_set/grid_s065.png`.

### Experiment 2: LoRA fine-tune

In progress. The fine-tuned model is evaluated on the same six scenes, seeds and arms as Experiment 1, prompted with the trained token `sks woman`.

## What we learned

**CLIP reports success on a consistency task that is failing.** In the baseline, CLIP-I self-consistency was about 0.80 while DINOv2 was about 0.24 on the same images. CLIP was trained to match images to semantic descriptions, so six portraits of auburn-haired, freckled women score as near-identical. DINOv2 is instance-discriminative and correctly treats them as different people. A category-level metric alone would have hidden the problem.

**Seed and guidance do not control identity.** Fixing the seed moved DINO from 0.24 to 0.29, and raising guidance to 12 added nothing further. With 15 pairs per arm, gaps this size are within noise. Both fixed-seed arms also followed the scene slightly worse (CLIP-T 0.233 down to about 0.21), consistent with being anchored to one initial noise sample.

**Text conditioning produces an attribute bundle, not a person.** Every baseline image kept the high-prior attributes: red curly hair, freckles, green eyes. Face shape, apparent age and hair length drifted between scenes. The most distinctive attribute in the description, a small scar above the left eyebrow, was absent from all six scenes and from the dedicated hero portrait.

**Img2img from one frontal reference cannot produce pose diversity.** The variation prompts asked for profile and three-quarter views, different lighting and darker backgrounds. None of these appeared. Raising strength from 0.5 to 0.65 lowered DINO self-consistency only from 0.971 to 0.950. Img2img starts from the reference's noised latent, so spatial layout survives at any strength that also preserves the face. A 0.97 training set is a warning sign, not a success: it means near-duplicates.

**Shared clothing and background risk entanglement.** Every training image shares the sage green sweater and the grey backdrop, so the LoRA could learn `sks woman` as "a woman in a green sweater". The training run names both in the instance prompt, so the model can attribute them to those words rather than to the token. Whether green fabric leaks into the armor and spacesuit scenes is the check.

## Method

**Character:** a 28-year-old woman with short curly auburn hair, light freckles across the nose, green eyes, and a small scar above the left eyebrow.

**Scenes:** coffee shop, rainy street at night, mountain trail, space station in a spacesuit, stone courtyard in medieval armor, summer picnic.

**Arms:** `prompt_only` (varied seed, CFG 7), `fixed_seed` (one seed, CFG 7), `high_cfg` (one seed, CFG 12). SDXL base, 30 steps, 1024 x 1024.

**Metrics:**

| Metric | Model | What it measures |
| --- | --- | --- |
| CLIP-I self-consistency | CLIP ViT-L/14 image embeddings | Semantic similarity between generations |
| DINO self-consistency | DINOv2-base CLS token | Instance-level similarity between generations |
| CLIP-T scene fidelity | CLIP ViT-L/14 image vs scene text | Whether the scene was followed |

**Training:** DreamBooth LoRA on the SDXL UNet using the official diffusers script. Rank 16, learning rate 1e-4 constant, 800 steps at batch size 1, text encoders frozen, no prior preservation.

## Repository layout

```
eval/consistency_experiment.py   Generation and scoring harness: baseline arms, LoRA arms,
                                 single-folder diagnostics, and before/after comparison
scripts/make_character_set.py    Builds the synthetic training set (hero + img2img variations)
scripts/train_lora.sh            Training config for 24GB cards with bf16 (L4, A10, A100)
scripts/train_lora_t4.sh         Training config for 16GB T4 cards
notebook/run_experiments_t4.ipynb  Colab runner for the full pipeline (runs on T4 or L4)
data/                            Training images (gitignored), see data/README.md
results/                         Metrics, JSON and contact sheets (tracked)
RESULTS.md                       Detailed write-up of each experiment
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

**1. Baseline:**

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

**4. Evaluate the fine-tuned model and compare:**

```bash
python eval/consistency_experiment.py --out_dir results/lora --lora_dir outputs/lora --character "sks woman"
python eval/consistency_experiment.py --compare results/baseline/eval.json results/lora/eval.json
```

`--lora_scale` sets adapter strength, and `--lora_dir outputs/lora/checkpoint-400` evaluates the intermediate checkpoint.

## Engineering notes

These came up while running on current Colab images and are handled in the code.

- **SDXL's stock VAE produces NaNs in fp16.** All pipelines load `madebyollin/sdxl-vae-fp16-fix`. On a T4, which has no bf16, this is what makes fp16 work at all.
- **diffusers 0.39 moved VAE slicing** from the pipeline to `pipe.vae.enable_slicing()`.
- **transformers 5 changed CLIP's `get_text_features` and `get_image_features`** to return a model output instead of a tensor. The harness calls the text and vision submodules and applies the projection explicitly. Using `pooler_output` directly would compare 768-dim text states against 1024-dim vision states outside the shared CLIP space.
- **Pipeline placement depends on available memory.** Cards with 20GB or more keep the whole pipeline on the GPU; smaller cards use CPU offload.
- **The training script is fetched for the installed diffusers version.** Scripts on the diffusers `main` branch usually require an unreleased dev version.
- **PEFT refuses to run with Colab's preinstalled torchao 0.10.** The project does not use torchao, so setup uninstalls it.
- **The `diffusers[training]` extra pins protobuf below 4**, which conflicts with Colab's Google libraries. Setup installs the needed packages individually and pins protobuf to a compatible range.
- **LoRA scale is set through `set_adapters`**, which works under the PEFT backend.
- **OpenCV 5 removed the Haar cascade face detector** from the main package. Face-cropped scoring needs the 4.x line, pinned in setup.

## Limitations

- **Small samples.** Six images per arm gives 15 pairs. Treat gaps below about 0.05 as inconclusive.
- **Whole-frame metrics.** DINO and CLIP-I score the entire image, so scene, color and composition affect similarity alongside identity. Before and after deltas remain valid because scenes and seeds are identical across runs, but absolute values understate identity similarity. Face-cropped scoring or ArcFace embeddings would isolate identity.
- **Synthetic, low-diversity training data.** All training images are frontal head-and-shoulders portraits with the same clothing and background. The fine-tuned model is likely to favor that framing.
- **No reference ground truth for Experiment 1.** The character exists only as a text description, so baseline identity is measured as self-consistency rather than similarity to a reference.
- **One base model, sampler and resolution.** No claims are made about other backbones.

## Next steps

- Complete Experiment 2 and report before and after results on all three arms.
- Add face-cropped identity scoring with ArcFace.
- Sweep LoRA scale and compare the step-400 and step-800 checkpoints to measure the identity versus editability tradeoff.
- Build a more diverse training set with multi-view generation or pose conditioning.
