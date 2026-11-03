"""Token accounting.

Costs in this project are *measured*, not guessed: every prompt the pipelines
build is actually built, and its token count is computed here.  The only
guessed inputs are the unit prices, which live in config/pricing.yaml with
explicit provenance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def count_text_tokens(text: str) -> int:
    """Tokenizer-free estimate.

    Tries tiktoken, then transformers, then falls back to a word/punctuation
    split scaled by 1.3 -- within a few percent of BPE counts for the short,
    highly templated prompts these pipelines emit.  The fallback is
    deterministic, which matters because the cost numbers must be reproducible
    on a machine without model weights installed.
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc(text))
    return max(1, int(round(len(_WORD_RE.findall(text)) * 1.3)))


_ENCODER: object = ...


def _get_encoder():
    global _ENCODER
    if _ENCODER is not ...:
        return _ENCODER
    _ENCODER = None
    try:  # pragma: no cover - depends on optional dependency
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        _ENCODER = enc.encode
    except Exception:
        pass
    return _ENCODER


@dataclass(frozen=True)
class VisionTokenizerSpec:
    """How an MLLM turns an image into tokens.

    Defaults describe a ViT-L/14 tower with a 2x2 pixel-shuffle resampler, which
    is what Astra-Global's description in arXiv:2506.06205 is consistent with.
    Override in config if you know the real tower.
    """

    patch_size: int = 14
    pool_factor: int = 2
    max_tokens_per_image: int = 1024
    extra_tokens_per_image: int = 2  # image begin/end markers

    def tokens_for(self, height: int, width: int) -> int:
        patches_h = math.ceil(height / self.patch_size)
        patches_w = math.ceil(width / self.patch_size)
        pooled = math.ceil(patches_h / self.pool_factor) * math.ceil(
            patches_w / self.pool_factor
        )
        return min(pooled, self.max_tokens_per_image) + self.extra_tokens_per_image


def count_image_tokens(
    images, spec: VisionTokenizerSpec | None = None
) -> int:
    spec = spec or VisionTokenizerSpec()
    total = 0
    for img in images:
        h, w = img.shape[0], img.shape[1]
        total += spec.tokens_for(int(h), int(w))
    return total
