"""Identity-aware evaluation metrics for character-consistent generation.

- CLIP-T           text<->image similarity  (prompt fidelity)
- CLIP-I           gen<->reference image similarity  (identity fidelity, semantic)
- DINO             gen<->reference image similarity  (identity fidelity, instance-level)
- self_consistency gen<->gen similarity across scenes  (is it the SAME character?)

All scores are mean cosine similarities on L2-normalized embeddings.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoModel,
    AutoImageProcessor,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class CLIPScorer:
    """Wraps a CLIP model for image and text embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str | None = None):
        self.device = device or _device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def image_embeds(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)

    @torch.no_grad()
    def text_embeds(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        emb = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(emb, dim=-1)


class DINOScorer:
    """DINO ViT embeddings (CLS token), sensitive to instance-level identity."""

    def __init__(self, model_name: str = "facebook/dino-vitb16",
                 device: str | None = None):
        self.device = device or _device()
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def image_embeds(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        cls = out.last_hidden_state[:, 0]  # CLS token
        return torch.nn.functional.normalize(cls, dim=-1)


def _mean_pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine similarity over the full cross-product of a x b."""
    sims = a @ b.T  # already normalized -> dot product is cosine
    return sims.mean().item()


def clip_t(scorer: CLIPScorer, images: Sequence[Image.Image],
           prompts: Sequence[str]) -> float:
    """Prompt fidelity: each image vs its own prompt (paired)."""
    img = scorer.image_embeds(images)
    txt = scorer.text_embeds(prompts)
    paired = (img * txt).sum(dim=-1)  # diagonal cosine, image_i vs prompt_i
    return paired.mean().item()


def image_identity(scorer, gen_images: Sequence[Image.Image],
                   ref_images: Sequence[Image.Image]) -> float:
    """Identity fidelity: generated vs reference character images (CLIP-I or DINO)."""
    gen = scorer.image_embeds(gen_images)
    ref = scorer.image_embeds(ref_images)
    return _mean_pairwise_cosine(gen, ref)


def self_consistency(scorer, gen_images: Sequence[Image.Image]) -> float:
    """Do the generations look like the same character across scenes?

    Mean cosine over all unique generated-image pairs.
    """
    emb = scorer.image_embeds(gen_images)
    n = emb.shape[0]
    if n < 2:
        return float("nan")
    sims = []
    for i, j in itertools.combinations(range(n), 2):
        sims.append(torch.dot(emb[i], emb[j]).item())
    return sum(sims) / len(sims)
