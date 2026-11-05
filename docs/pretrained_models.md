# Pretrained tokenizers on Hugging Face

The trained NTT tokenizers are published at [`the-enigma-project/ntt-tokenizers`](https://huggingface.co/the-enigma-project/ntt-tokenizers). Each model is one folder holding `config.json` (the tokenizer spec) and `model.safetensors` (the weights); no pickles are involved. The LDNS-Perceiver folder also holds a NuCLR embeddings `.safetensors` file. Weights, embeddings and model cards are licensed CC-BY-NC-4.0.

Model folders have public names such as `conv-vae-100to1`. Pin a release with a revision such as `v1.0`.

## 1. Use a tokenizer

```python
import torch
from ntt.hub import load_pretrained

tok = load_pretrained("conv-vae-100to1", revision="v1.0")  # downloads once, then cached
spec = tok.spec
print(spec.n_neurons, spec.window_length_seconds, spec.bin_size_ms, spec.compress_n)
```

`load_pretrained` also accepts a local export folder (one that contains `config.json`) in place of a name.

A batch holds binned spike counts in `spike_matrix`, shaped `[batch, n_bins, n_units]` with `n_bins = window_length_seconds * 1000 / bin_size_ms`. `tok.spec` tells you the rest:

- `bin_size_ms` is the bin width in milliseconds.
- `window_length_seconds` is a number for most models. For some (the Transformer-VQ models) it is a distribution such as `UniformDist(low=0.1, high=2.0)`, the range of window lengths used in training; choose a length in that range whose `n_bins` is a multiple of `spec.patch_length`.
- `n_units`: with `n_neurons=False` (the LDNS-Perceiver) the model takes any number of units. With an integer `n_neurons` and `compress_n=True` the model reads exactly `n_neurons` units. With `compress_n=False` the evaluation passes all units of a recording to the model directly.
- If `input_type` is `binary_spikes` and `binarize_spikes` is true, ntt's data pipeline clips counts to 0/1 before the model sees them; do the same.

A batch can be built and encoded with

```python
from ntt.core.model.tokenizer.validation import make_dummy_input

# Pass window_length_seconds=... when the spec's window is a distribution, and n_neurons=...
# for the number of units you feed (needed when spec.n_neurons is False).
batch = make_dummy_input(spec, batch_size=2)
batch = batch._replace(spike_matrix=my_counts)  # int32, same shape as batch.spike_matrix
with torch.no_grad():
    latents = tok.encode(batch)
```

**LDNS-Perceiver:** `make_dummy_input` sets a placeholder `session_id`, for which the perceiver uses all-zero unit embeddings: the outputs are meaningless and no error is raised. Set `session_id` to a session in the uploaded embeddings table and `unit_ids` to the units' indices within that session's units table (as ntt's data pipeline numbers them):

```python
import json
from safetensors import safe_open

with safe_open(tok.spec.embeddings_path, framework="pt") as f:
    sessions = json.loads(f.metadata()["sessions"])  # the sessions the table covers

batch = make_dummy_input(tok.spec, batch_size=2, n_neurons=n_units)
batch = batch._replace(
    spike_matrix=my_counts,  # [2, n_bins, n_units]
    session_id=(my_session,) * 2,  # one of `sessions`
    unit_ids=torch.as_tensor(my_unit_indices, dtype=torch.int32).repeat(2, 1),  # [2, n_units]
)
```

For recordings with more units than a fixed-`n_neurons` model reads, the evaluation (`job.evaluate_all_neurons=true`, the default) first caps each session at `max_neurons` units with a seeded random subset (`select_eval_units` in `ntt.core.data.components.transforms`; the setting is `max_neurons` in `ntt.evaluation.config.schema.EvaluationConfig`), then, for `compress_n=True` models, splits the units into groups of `n_neurons` and runs the model on each group (`NeuronCoveringWrapper` in `ntt.evaluation.utils.neuron_covering`).

## 2. Re-run our evaluation

Each published model was scored on six evaluations, all on the **test** split: reconstruction, repeat SNR, input- and latent-corruption robustness, encoding and decoding. Their results are aggregated into seven scores per model. `scripts/eval/published_commands.py` writes the ntt commands that reproduce them, with the settings each model was scored with:

```bash
uv run python scripts/eval/published_commands.py conv-vae-100to1 \
    --data-dir /absolute/path/to/brainsets/processed --revision v1.0
```

It prints the commands; `--run` runs them in order. `--revision` is a tag or a commit (default `v1.0`), not the branch `main`, whose files can change under the same name; results land in `results/<revision>/` unless `--output-dir` says otherwise. `--axes` picks evaluations (`recon snr m1 m2 encoding decoding`, default all), and `all` in place of a model name means every model.

| Evaluation | Command | Results |
|---|---|---|
| reconstruction, repeat SNR, input/latent robustness | `job=published_eval`, or `job=published_eval_<model>` when that file exists | `results/<revision>/<model>/eval`, `eval_snr`, `eval_m1`, `eval_m2` |
| encoding | `job=downstream_encoding` with the model's recipe | `results/<revision>/<model>/encoding` |
| decoding | `job=downstream_decoding job.mode=benchmark` | `downstream/experiments_benchmark/ntt_downstream_<model>-<hash>/` |

The settings are config files you can read and reuse directly:

- `config/job/published_eval.yaml`: the shared settings of the four intrinsic evaluations. `config/job/published_eval_<model>.yaml` adds a model's own settings: its scoring window, the units its robustness is measured on, and the units its compression is computed on.
- `config/downstream/encoding/published_<model>.yaml`: a model's encoding recipe, when it differs from the shared recipe of its kind.
- `scripts/eval/published_protocol.yaml`: for each model, its score table, its encoding recipe and loss, and its decoding settings.

An `hf://<owner>/<repo>/<name>[@<revision>]` path works as `job.ckpt_path` for `job=eval`, `job=eval_all` (all its passes) and both robustness jobs, and as `job.ckpt` for the downstream jobs (`job=downstream_encoding`, and `job=downstream_decoding` in all its modes; in benchmark mode it benchmarks that one tokenizer, see [`downstream_eval.md`](downstream_eval.md)). The model is downloaded into the Hugging Face cache and converted to a local checkpoint under `$NTT_HUB_CACHE` (default `~/.cache/ntt/hub`).

Requirements:

- The public brainsets prepared with brainsets; see [`public_dataset_creation.md`](public_dataset_creation.md). `--data-dir` is the folder holding them.
- The downstream submodule, for encoding and decoding: `git submodule update --init downstream`.
- A GPU for encoding and decoding; decoding needs an Ampere or newer GPU (A100, RTX 30xx/40xx, L40S), because it uses flash-attn, and a Linux with glibc 2.32 or newer (e.g. Ubuntu 22.04; on older systems such as RHEL 8, flash-attn fails to import with ``GLIBC_2.32' not found``, so run decoding in a newer container). What our runs took: reconstruction about 1 hour and 20–40 GB of memory on 24 CPU cores; repeat SNR usually about 10 GB of memory, up to about 100 GB (Transformer-VQ); encoding a few hours on one GPU (LDNS-Perceiver: about 33 hours); decoding a few hours on one GPU (LDNS-Perceiver: about 20 hours on four GPUs).
- No MLflow server: encoding and decoding run with MLflow logging off, even when `MLFLOW_TRACKING_URI` is set. With `--mlflow`, both log to the server in `MLFLOW_TRACKING_URI`.

Reconstruction metrics are compared with the published Poisson references, which the configs load from the Hub (`poisson_references/` in the repository; its README describes them). They are computed on 0.1 s windows, whatever window a model is scored on, so the two structure-preservation metrics use the baseline values of 0.1 s windows. Without a reference (`job.poisson_reference=null`), the evaluation computes a Poisson baseline in the same run, on the model's own windows; scores then differ slightly from the published ones.

**Aggregation.** When the evaluations are done, write the aggregation input and build the tables, one per score table (models with 1000 ms tokens are scored in their own table):

```bash
uv run python scripts/eval/published_commands.py all --revision v1.0 --tracker results/v1.0
uv run python scripts/analysis/build_aggregation_tables.py --sheet results/v1.0/tracker_200ms.csv \
    --outdir results/v1.0/scores_200ms --gen-split test --common-parts-only
uv run python scripts/analysis/build_aggregation_tables.py --sheet results/v1.0/tracker_1000ms.csv \
    --outdir results/v1.0/scores_1000ms --gen-split test --common-parts-only
```

The tracker files hold the models whose results exist in `results/v1.0/`, with the decoding results of `--revision`. With your own `--output-dir`, use one per revision: the folders under it are not named after the revision. See [`eval_aggregation.md`](eval_aggregation.md) for the outputs and the seven scores. Each score file records the `comparability_version` of the rules that scored it, and the published scores carry the current one.

## 3. LDNS-Perceiver and NuCLR embeddings

The LDNS-Perceiver reads a NuCLR embedding for every unit it sees, from the embeddings table uploaded next to it. The uploaded perceiver supports only the sessions in that table, which are sessions of the public brainsets; on any other session it raises a `KeyError` naming the session.

It cannot be extended to other recordings by adding embeddings: a newly trained NuCLR encoder embeds units in a different space from the one the uploaded perceiver was trained on, and the NuCLR encoder behind the uploaded table is not released. For other recordings, train NuCLR, compute embeddings and train your own perceiver on them. ntt has the code for each step:

1. Train NuCLR: `uv run ntt job=nuclr_contrastive paths.run_name=<name>` (code in `ntt.contrib.nuclr`).
2. Compute the per-unit embeddings: `uv run python -m ntt.contrib.nuclr.eval.extract_embeddings --ckpt-path <nuclr.ckpt> --data-root <brainsets root> --data-config <config> --split <split> --output-path embeddings.pt`.
3. Optionally, save them as safetensors (the perceiver reads either format):

   ```python
   import torch
   from ntt.contrib.tokenizer.perceiver.perceiver_tokenizer import save_embeddings_safetensors

   result = torch.load("embeddings.pt", weights_only=False)  # your own file (holds numpy arrays); keys: embs, unit_ids, session_ids
   save_embeddings_safetensors("my_embeddings.safetensors", result["embs"], result["unit_ids"], result["session_ids"])
   ```

4. Train a perceiver with `uv run ntt job=train paths.run_name=<name> tokenizer@job.tokenizer_config=<your config> loss@job.loss_config=contrib/poisson_nll`. Your tokenizer config (a YAML file under `config/tokenizer/`) has `_target_: ntt.contrib.tokenizer.perceiver.perceiver_tokenizer.PerceiverTokenizerSpec` and `embeddings_path` set to your embeddings file; the `spec` in the uploaded model's `config.json` lists the settings we used.
