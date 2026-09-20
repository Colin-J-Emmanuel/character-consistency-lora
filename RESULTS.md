# Results

## Experiment 1: how far does prompting alone get you?

Before fine-tuning anything, I measured the problem. One character description,
six scenes, SDXL base, three arms varying only seed strategy and guidance scale.
Identity is measured as mean pairwise similarity between the six generations
(CLIP image embeddings and DINOv2 CLS embeddings). Scene fidelity is CLIP
similarity to the scene text only, with the character description excluded so
that identity and prompt-following stay separable.

Run: `python eval/baseline_consistency.py --out_dir results/baseline`

| Arm | Seed | CFG | DINO self-consistency | DINO worst pair | CLIP-I self-consistency | CLIP-T scene |
|---|---|---|---|---|---|---|
| prompt_only | varied | 7.0 | | | | |
| fixed_seed | fixed | 7.0 | | | | |
| high_cfg | fixed | 12.0 | | | | |

Grids: `results/baseline/grid_*.png`. Raw numbers: `results/baseline/baseline_eval.json`.

### What I observed

<!-- Fill these in from the actual run. Three or four honest sentences beats
     a page of hedging. Things to look for:
     - Does a fixed seed actually preserve identity when the scene prompt
       changes, or does it only stabilize composition and palette?
     - Does DINO separate the arms more sharply than CLIP-I? If so, say why:
       CLIP was trained on semantic categories and will score two different
       people in similar clothing as highly similar, DINOv2 is
       instance-discriminative.
     - What does raising guidance do to scene fidelity versus identity?
     - Which pair scored worst, and what changed between those two images?
       Hair length? Face shape? Age? Name the specific failure. -->

### Why this is the right baseline

Any fine-tuning result has to beat these numbers on the same metric
definitions, same character, same scenes, same seeds. Reporting a DINO score
for a LoRA without a measured floor to compare it against says nothing.

## Experiment 2: LoRA fine-tune

<!-- To be filled after the training run. Reuse the same six scenes and the
     same seeds so the comparison is apples to apples. -->

## Limitations

- Six scenes and six images per arm is a small sample. Pairwise similarity
  means are noisy at this N, so treat gaps below roughly 0.05 as inconclusive.
- The character is synthetic, generated from a text description rather than
  photographed, so there is no ground-truth reference set for this experiment.
  Reference-based CLIP-I and DINO enter in Experiment 2.
- Single base model, single sampler, single resolution. No claim is made about
  how these numbers transfer to other backbones.
