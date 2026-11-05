# Model Glossary

What each model family is, in plain terms. Families are referenced by their short name
(VAE, BSQ, ...). Only the families shown in this view are listed.

Most families — VAE, MMD-VAE, BSQ, FSQ, VQ-PT, Diffusion-Prior — share one
**convolutional autoencoder** (it processes each neuron on its own and compresses a 100 ms
window into a single token (VAE-B: a 1 s window); they differ only in how the
latent code is constrained.

- **VAE** (published as `conv-vae-*`) — a continuous latent, lightly pulled toward a standard normal.
- **MMD-VAE** — the same, but it shapes the overall spread of latents (not each one
  individually) to avoid the latent collapsing.
- **BSQ** — a discrete latent where each dimension becomes a single bit (a corner of a
  binary cube).
- **FSQ** — a discrete latent where each dimension is rounded to a few fixed levels (a
  simple grid of codes, with no learned codebook).
- **VQ-PT** — a discrete latent built from off-the-shelf vector quantizers, here stacked
  in residual stages for finer codes.
- **Diffusion-Prior** — a continuous latent whose prior is a small learned diffusion model
  rather than a fixed bell curve.
- **Compressed Sensing** — a non-learned baseline: project each neuron's activity through a
  fixed random matrix, then reconstruct it by assuming the signal is sparse.
- **CaLM** (published as `transformer-vq-*`) — a per-neuron Transformer that turns each neuron's trace into discrete tokens;
  it learns mainly by reconstructing the trace, with a small extra nudge to make the tokens
  predictable in sequence (like a language model).
- **Cosmos** (published as `wavelet-*`) — treats the neurons × time array like an image and compresses both axes at
  once (all neurons of a 1 s window become a short sequence of tokens).
- **PCA** — a linear, non-learned baseline that keeps the top 1 or 10 principal components
  of each neuron's 1 s window.
- **LDNS** — a population autoencoder built from state-space layers; each token covers
  10 ms of the whole population. The Perceiver variant handles any number of neurons.
- **VAE-B** (published as `conv-vae-1000to1`) — the VAE's autoencoder, made deeper, compressing a whole 1 s window
  of each neuron into a single token.
