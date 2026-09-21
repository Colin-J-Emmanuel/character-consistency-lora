# Character dataset

Training images go in `my_character/`, which is gitignored. A contact sheet of the set used for each run is saved to `results/training_set/` so the repo still shows what was trained on.

## Building the set

The default path is synthetic: one hero portrait plus img2img variations of it.

```bash
python scripts/make_character_set.py --out_dir data/my_character --strength 0.65
```

`--strength` trades variety against identity. Lower values give near-duplicates; higher values drift the face. Img2img preserves the hero's spatial layout, so it cannot produce new head poses from a single frontal portrait at any strength.

## Checking it before training

```bash
python eval/consistency_experiment.py --eval_dir data/my_character
```

This reports self-consistency across the set. A LoRA cannot be more consistent than its training data, and a very high score (around 0.97 DINO) means the images are near-duplicates. Look at the images too, and remove any where the face is visibly a different person.

## Guidelines for your own images

- 8 to 15 images of one character, the same identity in each.
- Vary pose, angle, expression and lighting.
- Vary clothing and background where possible. If they cannot vary, name them in the instance prompt so the model does not bind them to the character token.
- Crop reasonably tight to the subject and avoid other prominent people.
- Square crops. `train_lora.sh` trains at 1024, `train_lora_t4.sh` at 768.
- Use only images you have the rights to, and document the source in the top-level README.
