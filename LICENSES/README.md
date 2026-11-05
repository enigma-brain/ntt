# Third-party licences

The code in this repository is MIT-licensed ([`../LICENSE`](../LICENSE)). The files
below adapt third-party code and keep its licence; each file's header names its
upstream. This folder holds the upstream licence texts, copied verbatim from each
upstream repository. Where a text names a copyright holder it is given below; the
Apache-2.0 texts of several upstreams carry only the unfilled template line, so no
holder is stated for them.

| Licence file | Upstream | Adapted in |
|---|---|---|
| [`nuclr-Apache-2.0.txt`](nuclr-Apache-2.0.txt) | [NuCLR](https://github.com/nerdslab/nuclr) (Arora et al., 2025) | `src/ntt/contrib/nuclr/**` (`contrastive_loss`, `distributed`, `embedding_collector`, `encoder`, `encoder_binned`, `litmodule`, `eval/labels`, `eval/probe`, `nn/attention`, `nn/ffn`, `nn/rotary_embedding`, `nn/sine_emb`) |
| [`torchvision-BSD-3-Clause.txt`](torchvision-BSD-3-Clause.txt) | [torchvision](https://github.com/pytorch/vision), Copyright (c) Soumith Chintala 2016 | `src/ntt/contrib/loss/focal.py` |
| [`fvcore-Apache-2.0.txt`](fvcore-Apache-2.0.txt) | [fvcore](https://github.com/facebookresearch/fvcore), Copyright 2019, Facebook, Inc | `src/ntt/contrib/loss/focal.py` (original of the torchvision port) |
| [`vector-quantize-pytorch-MIT.txt`](vector-quantize-pytorch-MIT.txt) | [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), Copyright (c) 2020 Phil Wang | `src/ntt/contrib/tokenizer/fsqffn/{__init__,quantizer}.py`, `wanneural/bottleneck.py`, `cosmos/bottleneck.py` |
| [`ldns-MIT.txt`](ldns-MIT.txt) | [LDNS](https://github.com/mackelab/LDNS), Copyright (c) 2024 mackelab | `src/ntt/contrib/tokenizer/ldns/{__init__,blocks,ldns_tokenizer,spike_adapter,count_wrapper}.py` |
| [`s4-Apache-2.0.txt`](s4-Apache-2.0.txt) | [S4](https://github.com/state-spaces/s4) (Gu, Goel and Re, 2022) | `src/ntt/contrib/tokenizer/ldns/s4.py` |
| [`wan-Apache-2.0.txt`](wan-Apache-2.0.txt) | [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2) (identical licence texts) | `src/ntt/contrib/tokenizer/wanneural/{blocks,encoder,decoder,bottleneck,wanneural}.py` |
| [`diffusers-Apache-2.0.txt`](diffusers-Apache-2.0.txt) | [diffusers](https://github.com/huggingface/diffusers) | `src/ntt/contrib/tokenizer/wanneural/bottleneck.py` (`DiagonalGaussianDistribution`) |
| [`cosmos-tokenizer-Apache-2.0.txt`](cosmos-tokenizer-Apache-2.0.txt) | [Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) (NVIDIA) | `src/ntt/contrib/tokenizer/cosmos/{__init__,cosmos_2d,layers2d,patcher,bottleneck}.py` |
| [`torchtitan-BSD-3-Clause.txt`](torchtitan-BSD-3-Clause.txt) | [torchtitan](https://github.com/pytorch/torchtitan), (c) Meta Platforms, Inc. and affiliates | `src/ntt/core/model/components/lr_scheduler.py`, `src/ntt/contrib/tokenizer/fsqffn/{blocks,utils}.py` |
| [`torch_brain-Apache-2.0.txt`](torch_brain-Apache-2.0.txt) | [torch_brain](https://github.com/neuro-galaxy/torch_brain), Copyright 2025 NeuroGalaxy | `src/ntt/core/data/components/transforms.py` (`UnitDropout`) |
| [`phy-BSD-3-Clause.txt`](phy-BSD-3-Clause.txt) | [Phy](https://github.com/cortex-lab/phy), Copyright (c) 2019, Cyrille Rossant | `src/ntt/evaluation/core/intermediates.py` (`_compute_ccgs`) |
| [`spikeinterface-MIT.txt`](spikeinterface-MIT.txt) | [SpikeInterface](https://github.com/SpikeInterface/spikeinterface), Copyright (c) 2019 SpikeInterface | `src/ntt/evaluation/core/intermediates.py` (`_compute_ccgs`, via its adaptation of Phy) |
