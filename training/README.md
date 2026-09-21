# Training code

This is the fine-tuning code behind the `p_drop_conditionning: 0.1` runs
described in the [main write-up](../README.md) ("Change 2: classifier-free
guidance"), configured via
`ctflow/lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all_ft.yaml`.

It's extracted from a larger private research monorepo, not a standalone
package — cluster-specific values (home-directory paths, proxy settings,
workspace IDs) in `slurm/*.sh` and the config have been replaced with
placeholders, so treat these as illustrative rather than runnable
out-of-the-box; you'll need to point them at your own cluster/storage
layout. No weights, datasets, or container images are included here; the
trained checkpoint and inference code are on
[Hugging Face](https://huggingface.co/EnyaWoooo/ctflowv2-vlm3d2026).

## Layout

- `ctflow/lvfm/train_ft.py` — training loop: rectified-flow / velocity
  matching on VAE latents, autoregressive block conditioning on the
  previous latent block, text conditioning with independent CFG dropout
  for text and image conditions.
- `ctflow/lvfm/configs/` — the fine-tuning config referenced above.
- `ctflow/common/models.py` — the denoiser architecture (`DiffuserSTDiT` /
  `STDiT`), a spatio-temporal DiT-style transformer.
- `ctflow/common/datasets_ft.py` — `LatentBlockDatasetV2`: loads
  precomputed VAE-latent blocks, builds captions on the fly from CT-RATE
  report/metadata CSVs.
- `ctflow/common/schedulers.py`, `ctflow/common/__init__.py` — LR
  schedule and shared training utilities (latent sampling, noise, VAE
  scaling helpers).
- `slurm/` — the multi-node Slurm launch scripts used for these runs.
- `environment_v2.yaml`, `tmi_container_v2.def` — environment / container
  definition for reproducibility.
