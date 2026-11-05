# Metric Glossary

The headline metrics — what each measures and whether **higher** or **lower** is
better. The local dashboard (`scripts/analysis/dashboard.py`) shows the full per-metric reference in its dev view (the "dev mode" toggle, which lists every metric and tokenizer); see also `metric_glossary.md`.

## The 7 composite scores (radar axes)

Each summarizes a tokenizer, aggregated from many raw metrics and scaled so higher is
better.

- **SN Recon (Poisson-norm.)** — how well single-neuron spike-train statistics are
  reconstructed, scored against a rate-matched Poisson surrogate (0 = no better than
  Poisson, 1 = perfect). Firing-rate error is the exception: it is scored against giving
  every neuron the population's mean rate. **Higher is better.**
- **Population Structure (Poisson-norm.)** — the same, for population-level structure
  (cross-neuron correlations, effective dimensionality). **Higher is better.**
- **Decoding (POYO + linear)** — held-out R² of behavior decoders reading the
  frozen tokenizer's latents: the mean of **POYO decoding (cursor-velocity R²)**, a
  trained POYO decoder, and **Linear decoding (ridge R²)**, in-process ridge regressors.
  Both parts are also shown on their own. **Higher is better.**
- **Downstream Encoding (nc_r²)** — noise-ceiling-corrected R² of a stimulus→activity
  encoding model built on the latents. **Higher is better.**
- **Latent Structure & Robustness** — how well the latent preserves the data's
  structure and how stable it stays under perturbations. **Higher is better.**
- **Generalization** — how well reconstruction quality holds up on datasets unseen in
  training (new sources) vs the in-distribution baseline. **Higher is better.** The
  Generalization tab shows reconstruction metrics only, each tokenizer's tiers compared
  over the metrics all three tiers share; new sessions and new subjects of the training
  datasets are shown for reference but show no measurable loss, so they are not scored.
- **Compression** — input bit-budget ÷ latent bit-budget; > 1 means the latent carries
  fewer bits than the input. **Higher is better** — read alongside reconstruction, since
  a tokenizer that discards information also compresses well. On the radar it is drawn
  on a log scale relative to the lowest and highest rate of all models shown.

## Reconstruction metrics

- **firing rate error** (single unit) — absolute error in each neuron's mean firing
  rate (Hz). **Lower is better.**
- **acg correlation** (single unit) — correlation between original and reconstructed
  per-neuron auto-correlograms (a neuron's spike timing relative to itself).
  **Higher is better.**
- **spike detection f1** (temporal) — F1 for placing reconstructed spikes within ~1 ms
  of real spikes. **Higher is better.**
- **spike count correlation** (temporal) — correlation between original and
  reconstructed binned spike-count series. **Higher is better.**
- **population coupling preservation** (population) — correlation, across neurons, of
  each neuron's coupling to the rest of the population, original vs reconstruction.
  **Higher is better.**
- **participation ratio error** (population) — absolute error in the effective
  dimensionality of population activity. **Lower is better.**
- **population structure preservation** (latent) — RSA: whether the latent keeps
  windows that were similar in activity similar in latent space. **Higher is better.**

## Downstream metrics

- **encoding / nc_r2** — noise-ceiling-corrected R² (see Downstream Encoding above).
  **Higher is better.**
- **decoding / test_r2** — POYO decoding (cursor-velocity R²): held-out POYO decoder
  test R², one of the two parts of the Decoding composite. **Higher is better.**

## Reading the plots

- **Axis arrows** — **↑** higher-is-better, **↓** lower-is-better.
- **Values toggle** — **Raw** shows the metric's native value; **Poisson-norm** rescales
  reconstruction metrics so 0 = a rate-matched Poisson baseline and 1 = perfect
  (for firing-rate error, 0 = every neuron given the population's mean rate).
- **Ranking whiskers** show a metric's spread across recording sessions.
