# Model Glossary

What each tokenizer family is, in one or two sentences. Families are referenced by their
public Hugging Face name stem (e.g. `conv-vae-*`, `wavelet-*`), with the internal family
name used in code and configs given as an alias (e.g. `wanx_A_vae`, `cosmos`). Only
families shown in the local dashboard (`scripts/analysis/dashboard.py`) are listed.

## The convolutional family (`conv-*`, internal name `wanx_A_*`)

All `conv-*` models (internal `wanx_A_*`) share one architecture: a **2D (time x neuron) convolutional
autoencoder** adapted from the Wan video VAE, applied **per neuron** and compressing along
**time** (a 100 ms window becomes 1 token). They differ only in the **bottleneck** that
constrains the latent.

- **`conv-vae-*`** (internal `wanx_A_vae`) — continuous **VAE** latent: a diagonal-Gaussian code with a small
  (usually annealed) KL penalty toward a standard normal.
- **`conv-mmd-vae-*`** (internal `wanx_A_mmd`) — the same Gaussian head, but the per-sample KL is replaced by an
  **MMD / InfoVAE** penalty on the aggregate posterior, which fights posterior collapse
  and latent holes.
- **`conv-bsq-*`** (internal `wanx_A_bsq`) — **Binary Spherical Quantization**: the latent is L2-normalized and
  each dimension is sign-quantized, giving a binary-hypercube codebook (discrete), with an
  entropy penalty for code usage.
- **`conv-fsq-*`** (internal `wanx_A_fsq`) — **Finite Scalar Quantization**: each latent dimension is bounded and
  rounded to a few fixed integer levels (an implicit lattice codebook, discrete) — no
  learned codebook and no commitment loss.
- **`conv-rfsq-*` / `conv-rlfq-*`** (internal `wanx_A_vqpt`) — a pluggable **vector-quantize-pytorch** bottleneck (discrete); the
  shown variants are residual FSQ, residual lookup-free quantization, and single-stage FSQ.
- **`wanx_A_dprior`** (internal name; no `conv-*` model of this variant is published) — a **learned diffusion prior**: the encoder is deterministic and a
  small DiT-style diffusion model is trained jointly as the latent's prior, replacing the
  VAE's fixed-Gaussian KL.

## `wavelet-*` (internal name `cosmos`)

A **Cosmos-style 2D tokenizer**: it treats the (neuron x time) array as an image, applies
a **local wavelet patchify (non-causal only at the window boundary) and a
causal-convolution ResNet encoder/decoder**, and compresses **both the neuron and time
axes** (independently configurable) — the published sweep collapses all 32 neurons to one and
lightly compresses time. `lc2`/`lc4` set the latent capacity (2 vs 4 channels / quantizer stages); the
`ae`/`vae`/`fsq`/`rfsq` suffixes are a continuous autoencoder, a continuous VAE,
finite-scalar quantization, and residual FSQ.

## `cs-*` (internal name `compressed_sensing`)

A non-learned **compressed-sensing** baseline: encoding is a single fixed **random linear
projection** of each neuron's time series, and decoding is **L1 / basis-pursuit**
reconstruction. The `gaussian` and `sinusoidal` variants differ only in the sensing matrix
— i.i.d. Gaussian entries versus sin/cos rows at geometrically spaced frequencies.

## `transformer-vq-*` (internal name `calm`)

A **per-neuron Transformer** tokenizer: each neuron's trace is patchified and encoded by a
stack of RoPE, causally-masked Transformer blocks, then quantized to a shared discrete
vocabulary. It is trained **primarily as a reconstruction autoencoder**, with a small
**auxiliary next-token** regularizer (CaLM-style) on the quantized tokens. An optional
population-context stage mixes across neurons before quantization (`pop_context` vs
`no_context`).

## `conv-vae-1000to1` (internal name `wanx_B`)

The `wanx_A` (`conv-*`) per-neuron convolutional autoencoder with a deeper stack and a **VAE**
bottleneck, configured so that a whole **1 s window of each neuron becomes one token**.
Evaluated on 5 s windows; shown as squares (the 1000 ms-token cohort), and left out of
decoding (its behavior is decoded from 1 s chunks).

## `ldns` and `ldns-perceiver` (internal name `LDNS`)

The autoencoder of **LDNS** (Kapoor et al., 2024, "Latent diffusion for neural spiking
data"): **S4 structured state-space** blocks encode the binned spike counts of the whole
population into a continuous latent and decode firing rates; one token per **10 ms** bin.
`LDNS - Perceiver` swaps the fixed-population encoder for a **Perceiver-S4** that accepts
any number of neurons.

## `pca-*` (internal name `PCA`)

A **linear PCA baseline** with no learned weights: per-neuron window covariance is
accumulated over the first training epoch and eigendecomposed into a fixed projection;
encode and decode are the PCA transform and its inverse. It is non-causal (sees the whole
window), so it serves as an "oracle" reference. `PCA_1000_1` / `PCA_1000_10` keep 1 vs 10
components.
