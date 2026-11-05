# WanNeural Tokenizer Architecture

Convolutional encoder-decoder adapted from the Wan2.1/Wan2.2 video VAE, operating on 2D neural data (Time x Neurons). Supports both VAE (continuous) and FSQ (discrete) latent bottlenecks.

## High-Level Pipeline

```
Input (B, T, N)                         e.g. (B, 2000, 32)
       │
       ▼
┌─────────────┐
│  Unsqueeze   │  (B, 1, T, N)          (B, 1, 2000, 32)
└──────┬──────┘
       ▼
┌─────────────┐
│  Patchify    │  (B, P, T/P, N)        (B, 50, 40, 32)       P=temporal_patch_size
└──────┬──────┘
       ▼
┌─────────────┐
│   Encoder    │  (B, head_dim, Tc, N)  (B, 32|16, 10, 32)    head_dim = 2·z for VAE, z for FSQ
└──────┬──────┘
       ▼
┌─────────────┐
│  Bottleneck  │  (B, z, Tc, N)         (B, 16, 10, 32)       VAE: sample z ~ N(μ,σ²)
│  (VAE/FSQ)   │                                                FSQ: quantize to codebook
└──────┬──────┘
       ▼
┌─────────────┐
│   Decoder    │  (B, P, T/P, N)        (B, 50, 40, 32)
└──────┬──────┘
       ▼
┌─────────────┐
│ Unpatchify   │  (B, 1, T, N)          (B, 1, 2000, 32)
└──────┬──────┘
       ▼
┌─────────────┐
│  Softplus    │  (B, T, N)             (B, 2000, 32)          squeeze + output activation
└─────────────┘
```

**Temporal compression** = `patch_size × stride^(num_downsamples)` = 50 × 2² = **200×**
**Tokens per sample** = T_compressed × N = 10 × 32 = **320**

---

## Encoder (`Encoder2d`)

```
(B, P=50, T/P=40, N=32)
       │
       ▼
┌──────────────────────────────────────┐
│  CausalConv2d  (50→64, k=(3,1))     │  (B, 64, 40, 32)
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 0: 64→64, no downsample      │  (B, 64, 40, 32)
│  ┌─ 2× ResBlock(64→64, k=(3,1))     │
│  └─ [+Attention if in attn_levels]   │
│  + AvgDown shortcut (identity)       │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 1: 64→128, ↓2 temporal       │  (B, 128, 20, 32)
│  ┌─ 2× ResBlock(64→128, k=(3,1))    │
│  └─ [+Attention if in attn_levels]   │
│  TemporalDownsample (stride=2)       │
│  + AvgDown shortcut (64→128, ↓2)    │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 2: 128→256, ↓2 temporal      │  (B, 256, 10, 32)
│  ┌─ 2× ResBlock(128→256, k=(3,1))   │
│  └─ [+Attention if in attn_levels]   │
│  TemporalDownsample (stride=2)       │
│  + AvgDown shortcut (128→256, ↓2)   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 3: 256→256, no downsample    │  (B, 256, 10, 32)
│  ┌─ 2× ResBlock(256→256, k=(3,1))   │
│  └─ [+Attention if in attn_levels]   │
│  + AvgDown shortcut (identity)       │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Mid Block                           │
│  ResBlock(256→256) → Attention →     │  (B, 256, 10, 32)
│  ResBlock(256→256)                   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Head: RMSNorm → SiLU → Conv2d      │  (B, head_dim, 10, 32)
│  VAE:  256→32  (z_dim×2)            │
│  FSQ:  256→16  (z_dim)              │
└──────────────────────────────────────┘
```

---

## Bottleneck

### VAE Mode (`merged_large_vae_b1_2`)

```
Encoder output: (B, 32, 10, 32)
       │
  pre_conv 1×1 (32→32)
       │
  Split → μ (B, 16, 10, 32)
       → logσ² (B, 16, 10, 32)
       │
  Reparameterize: z = μ + ε·σ          (B, 16, 10, 32)
       │
  post_conv 1×1 (16→16)  ──→  decoder
       │
  aux_loss = β · KL(q(z|x) || N(0,I))   β=1.0
```

Latent: **continuous**, z_dim=16

### FSQ Mode (`merged_large_fsq_2`)

```
Encoder output: (B, 16, 10, 32)
       │
  Reshape → (B·10·32, 16)              per-position quantization
       │
  project_in: Linear(16→4)             project to codebook dim
       │
  bound → round_ste → normalize         quantize each of 4 dims
       │                                 levels = [8, 5, 5, 5]
  project_out: Linear(4→16)            project back to z_dim
       │
  Reshape → (B, 16, 10, 32)
```

Latent: **discrete**, codebook size = 8×5×5×5 = **1000**

---

## Decoder (`Decoder2d`)

Mirror of the encoder with reversed stage ordering.

```
(B, z_dim=16, 10, 32)
       │
       ▼
┌──────────────────────────────────────┐
│  CausalConv2d  (16→256, k=(3,1))    │  (B, 256, 10, 32)
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Mid Block                           │
│  ResBlock(256→256) → Attention →     │  (B, 256, 10, 32)
│  ResBlock(256→256)                   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 0: 256→256, no upsample      │  (B, 256, 10, 32)
│  ┌─ 3× ResBlock(256→256, k=(3,1))   │  (decoder gets +1 res block)
│  └─ [+Attention if in attn_levels]   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 1: 256→256, ↑2 temporal      │  (B, 256, 20, 32)
│  ┌─ 3× ResBlock(256→256, k=(3,1))   │
│  └─ [+Attention if in attn_levels]   │
│  TemporalUpsample (interp + conv)    │
│  + DupUp shortcut (repeat ↑2)       │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 2: 256→128, ↑2 temporal      │  (B, 128, 40, 32)
│  ┌─ 3× ResBlock(256→128, k=(3,1))   │
│  └─ [+Attention if in attn_levels]   │
│  TemporalUpsample (interp + conv)    │
│  + DupUp shortcut (256→128, ↑2)    │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Stage 3: 128→64, no upsample       │  (B, 64, 40, 32)
│  ┌─ 3× ResBlock(128→64, k=(3,1))    │
│  └─ [+Attention if in attn_levels]   │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Head: RMSNorm → SiLU → Conv2d      │  (B, 50, 40, 32)
│  64→50 (=patch_size)                 │
└──────────────────────────────────────┘
```

---

## Building Blocks

### ResidualBlock
```
x ──────────────────────────────┐
│                               │ (shortcut: identity or 1×1 proj)
RMSNorm → SiLU → CausalConv2d  │
RMSNorm → SiLU → CausalConv2d  │
│                               │
└───────── + ◄──────────────────┘
```

### AttentionBlock (cross-neuron)
Self-attention across the N=32 neuron positions within each time frame independently.

```
x: (B, C, T, N)
│
RMSNorm
│
QKV projection (1×1 conv)
│
Reshape → (B·T, heads, N, d)       attention across N neurons per time-frame
│
Scaled dot-product attention
│
Reshape → (B, C, T, N)
│
Output projection (zero-init)
│
+ residual
```

### CausalConv2d
Causal (left-only) padding along time, symmetric padding along neurons. Ensures no future information leaks into current time steps.

### Stage Shortcuts (Wan2.2-style)
- **Encoder**: AvgDown2D — average pooling + optional 1×1 projection
- **Decoder**: DupUp2D — repeat-interleave + optional 1×1 projection

---

## Config Variants

| Config | dim | z_dim | levels | dim_mult | attn_levels | bottleneck | β | compression |
|---|---|---|---|---|---|---|---|---|
| `merged_large_vae_b1_2` | 64 | 16 | 4 | (1,2,4,4) | [] | VAE | 1.0 | 200× |
| `merged_large_fsq_2` | 64 | 16 | 4 | (1,2,4,4) | [] | FSQ [8,5,5,5] | — | 200× |
| `merged_large_vae_b01` (v1) | 64 | 16 | 4 | (1,2,4,4) | [0,1] | VAE | 0.01 | 200× |
| `merged_small_vae_b01` | 32 | 4 | 4 | (1,2,2,2) | [0,1] | VAE | 0.01 | 200× |

**v1 vs v2**: v2 removes cross-neuron attention (`attn_levels=[]`), treating neurons independently except through the mid-block attention.

**Merged vs Independent**: Independent variants use k_t=1 (no temporal mixing in res blocks) and s_t_down=3 (larger stride), giving isolated per-neuron temporal processing.
