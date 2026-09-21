# Results

All numbers are mean pairwise cosine similarity across the six images in an arm (15 pairs), except CLIP-T, which is the mean over images. Gaps below about 0.05 are within noise at this sample size. Raw values are in each `results/*/eval.json` and `eval_face.json`.

## Conditions

| Condition | Directory | Prompt subject | Adapter |
| --- | --- | --- | --- |
| Text only | `results/baseline` | Full character description | None |
| LoRA token only | `results/lora` | `sks woman` | LoRA, scale 1.0 |
| Text + LoRA | `results/lora_desc` | `sks woman` followed by the full description | LoRA, scale 1.0 |

Scenes, seeds, arms, sampler, steps and resolution are identical across all three.

## Face-cropped scoring

Largest detected face per image, 30% margin. 6 of 6 faces detected in every arm of every condition.

**DINO self-consistency (mean / worst pair)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.503 / 0.216 | 0.552 / 0.284 | 0.617 / 0.411 |
| fixed_seed | 0.663 / 0.533 | 0.555 / 0.344 | 0.698 / 0.502 |
| high_cfg | 0.617 / 0.476 | 0.574 / 0.351 | 0.706 / 0.497 |

**CLIP-I self-consistency (mean)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.857 | 0.798 | 0.866 |
| fixed_seed | 0.862 | 0.830 | 0.864 |
| high_cfg | 0.855 | 0.820 | 0.849 |

## Whole-frame scoring

**DINO self-consistency (mean / worst pair)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.239 / 0.060 | 0.162 / 0.005 | 0.362 / 0.096 |
| fixed_seed | 0.291 / 0.130 | 0.245 / 0.076 | 0.366 / 0.148 |
| high_cfg | 0.302 / 0.114 | 0.252 / 0.074 | 0.368 / 0.180 |

**CLIP-I self-consistency (mean)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.800 | 0.739 | 0.762 |
| fixed_seed | 0.822 | 0.772 | 0.817 |
| high_cfg | 0.795 | 0.767 | 0.803 |

**CLIP-T scene fidelity (mean)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.233 | 0.239 | 0.208 |
| fixed_seed | 0.209 | 0.232 | 0.205 |
| high_cfg | 0.213 | 0.236 | 0.217 |

## Training set

One txt2img hero portrait (seed 7) plus eleven img2img variations, scored on whole frames.

| img2img strength | DINO mean | DINO worst pair | CLIP-I mean |
| --- | --- | --- | --- |
| 0.50 | 0.971 | 0.942 | 0.975 |
| 0.65 (used) | 0.950 | 0.884 | 0.965 |

Grid: `results/training_set/grid_s065.png`.

## Training run

800 steps at batch size 1 (67 epochs over 12 images), rank 16, learning rate 1e-4 constant, 1024px, bf16, on an L4. About 1.4 seconds per step, 19.5 minutes total. Checkpoints saved at steps 400 and 800. Instance prompt: `a photo of sks woman, sage green sweater, plain grey backdrop`.

## Notes

- **Text + LoRA vs text only** is the controlled comparison. Face DINO rises by 0.11 (`prompt_only`) and 0.09 (`high_cfg`), and by 0.04 (`fixed_seed`), which is within noise.
- **The LoRA token alone underperforms the description** on face crops in all three arms.
- **The token-only condition has the highest scene fidelity** in every arm, consistent with a short subject leaving the scene words more weight. The same shift toward wider compositions explains its low whole-frame similarity.
- **Face-crop CLIP-I barely separates the conditions** (0.80 to 0.87), while face-crop DINO ranges from 0.50 to 0.71.
- **Green outerwear** appears in the coffee shop, rain and hiking scenes in both the text-only and token-only `prompt_only` grids, so it cannot be attributed to entanglement with the training sweater.
- **The scar** never appears in the training set and so could not be learned.
