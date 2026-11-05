"""Core neural network modules for the NuCLR model family.

Attention modules use ``torch.nn.attention.flex_attention`` (compiled
via ``torch.compile``) with a ``BlockMask`` to enforce per-segment
attention isolation.

Used by both the contrastive encoder (contrib/nuclr) and the
reconstruction tokenizer (contrib/tokenizer/nuclr_rec).
"""

from .attention import RotaryCrossAttention, RotarySelfAttention
from .ffn import FFN
from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb, invert_rotary_pos_emb
from .sine_emb import SinusoidalTimeEmb

__all__ = [
    "FFN",
    "RotaryCrossAttention",
    "RotaryEmbedding",
    "RotarySelfAttention",
    "SinusoidalTimeEmb",
    "apply_rotary_pos_emb",
    "invert_rotary_pos_emb",
]
