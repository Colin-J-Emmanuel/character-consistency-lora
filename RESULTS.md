# Results

All numbers are mean pairwise cosine similarity across the six images in an arm (15 pairs), except CLIP-T, which is the mean over images. Gaps below about 0.05 are within noise at this sample size. Raw values are in each `results/*/eval.json` and `eval_face.json`.

## Conditions

| Condition | Directory | Prompt subject | Adapter |
| --- | --- | --- | --- |
| Text only | `results/baseline` | Full character description | None |
| LoRA token only | `results/lora` | `sks woman` | LoRA, scale 1.0 |
| Text + LoRA | `results/lora_desc` | `sks woman` followed by the full description | LoRA, scale 1.0 |

Scenes, seeds, arms, sampler, steps and resolution are identical across all three.

## Face-recognition identity scoring (headline metric)

SFace embeddings on YuNet-detected, aligned faces. 6 of 6 faces detected in every arm of every condition.

**SFace identity similarity (mean / worst pair)**

| Arm | Text only | LoRA token only | Text + LoRA |
| --- | --- | --- | --- |
| prompt_only | 0.525 / 0.410 | 0.613 / 0.431 | 0.751 / 0.653 |
| fixed_seed | 0.545 / 0.375 | 0.679 / 0.546 | 0.809 / 0.695 |
| high_cfg | 0.648 / 0.509 | 0.675 / 0.530 | 0.798 / 0.633 |

**Change from text only to text + LoRA**

| Arm | Mean | Worst pair |
| --- | --- | --- |
| prompt_only | +0.226 | +0.243 |
| fixed_seed | +0.264 | +0.320 |
| high_cfg | +0.150 | +0.124 |

Every pair in every condition, including the baseline, exceeds OpenCV's 0.363 same-identity threshold, so that threshold separates nothing here and is reported only for completeness.

Training set (`data/training_set_s065`, 12 images): mean 0.722, worst pair 0.620, 12 of 12 faces detected. Text + LoRA with a fixed seed scores above this.

## Face-cropped scoring (DINOv2 and CLIP)

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

## Adapter scale sweep and checkpoint comparison

All runs use the text + LoRA prompt, the same scenes and seeds, and the adapter from the 800-step run unless stated. Scale 0.0 is the text-only baseline; scale 1.0 is `results/lora_desc`.

![LoRA scale sweep](scale_sweep.png)

**SFace identity (mean) vs LoRA scale**

| Arm | 0.0 | 0.4 | 0.7 | 1.0 |
| --- | --- | --- | --- | --- |
| prompt_only | 0.525 | 0.582 | 0.676 | 0.751 |
| fixed_seed | 0.545 | 0.662 | 0.748 | 0.809 |
| high_cfg | 0.648 | 0.721 | 0.752 | 0.798 |

**CLIP-T scene fidelity (mean) vs LoRA scale**

| Arm | 0.0 | 0.4 | 0.7 | 1.0 |
| --- | --- | --- | --- | --- |
| prompt_only | 0.233 | 0.233 | 0.218 | 0.208 |
| fixed_seed | 0.209 | 0.207 | 0.214 | 0.205 |
| high_cfg | 0.213 | 0.208 | 0.213 | 0.217 |

Identity rises monotonically in every arm with no sign of flattening at 1.0. Scene fidelity falls only in `prompt_only`, by 0.025 across the whole sweep, and is flat elsewhere. Either the identity-versus-editability crossover lies above scale 1.0, or CLIP-T is too coarse to capture the kind of editability loss that matters here, such as training-set clothing appearing in unrelated scenes. This run cannot distinguish the two.

**Step 400 vs step 800, both at scale 1.0**

| Metric | Arm | ckpt-400 | ckpt-800 |
| --- | --- | --- | --- |
| SFace identity | prompt_only | 0.690 | 0.751 |
| SFace identity | fixed_seed | 0.787 | 0.809 |
| SFace identity | high_cfg | 0.771 | 0.798 |
| CLIP-T scene | prompt_only | 0.211 | 0.208 |
| CLIP-T scene | fixed_seed | 0.206 | 0.205 |
| CLIP-T scene | high_cfg | 0.212 | 0.217 |

800 steps is better on identity in every arm with no measured cost to scene fidelity, so this configuration is not overtrained at 800. Separately, ckpt-400 at scale 1.0 scores close to the 800-step adapter at scale 0.7, which is consistent with both changes shrinking the effective magnitude of the weight update; that is a hypothesis this data suggests rather than tests.

Directories: `results/scale_04`, `results/scale_07`, `results/ckpt400`. Figure: `results/scale_sweep.png`, regenerated with `python eval/plot_scale_sweep.py`.

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

- **Text + LoRA vs text only** is the controlled comparison. SFace identity rises by 0.15 to 0.26 in all three arms, and the worst pair by 0.12 to 0.32. Face-cropped DINOv2 shows the same direction but smaller: +0.11, +0.09, and +0.04, the last within noise.
- **The LoRA token alone beats the description on SFace** in all three arms, but underperforms it on face-cropped DINOv2. DINOv2 still responds to framing, hair and expression after cropping, and the token-only generations vary more in composition; SFace aligns the face and scores identity alone, so it is the metric to trust for this question.
- **The adapter exceeds its training data's own consistency** (0.809 vs 0.722), so a LoRA is not capped by the coherence of its examples.
- **The token-only condition has the highest scene fidelity** in every arm, consistent with a short subject leaving the scene words more weight. The same shift toward wider compositions explains its low whole-frame similarity.
- **Face-crop CLIP-I barely separates the conditions** (0.80 to 0.87), while face-crop DINO ranges from 0.50 to 0.71.
- **Green outerwear** appears in the coffee shop, rain and hiking scenes in both the text-only and token-only `prompt_only` grids, so it cannot be attributed to entanglement with the training sweater.
- **The scar** never appears in the training set and so could not be learned.
