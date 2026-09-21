import argparse
import random
import importlib
import logging
import math
import os
import shutil
import types
import warnings
from copy import deepcopy
from functools import partial

import accelerate
import diffusers
import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import (check_min_version, deprecate, is_wandb_available,
                             make_image_grid)
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint_adjoint as odeint
from torchvision import transforms
from tqdm.auto import tqdm
import moviepy.editor as mpy
import imageio


import wandb
import transformers
if not hasattr(transformers, "deepspeed"):
    class _FakeDeepspeed:
        @staticmethod
        def is_deepspeed_zero3_enabled():
            return False
    transformers.deepspeed = _FakeDeepspeed()

from ctflow.common import *
from ctflow.common.schedulers import StepBasedLearningRateScheduleWithWarmup
#from ctflow.common.cached_datasets import instantiate_cached_dataset
#from ctflow.common.datasets_old import instantiate_dataset
from ctflow.common.datasets_ft import instantiate_dataset
#from ctflow.common.shared_memory import (custom_collate_fn,
#                                          instantiate_cached_dataset_sm)

"""

python ctflow/lvfm/train.py --config ctflow/lvfm/configs/lvfm_STDiT-S2_16f8_all.yaml

accelerate launch --num_processes 4 --multi_gpu --num_machines 1 --mixed_precision bf16 ctflow/lvfm/train.py --config ctflow/lvfm/cluster/lvfm_UNetSTIC-S_16f8_all.yaml

accelerate launch --num_processes 4 --dynamo_backend inductor --dynamo_use_dynamic --multi_gpu --num_machines 1 --mixed_precision bf16 ctflow/lvfm/train.py --config ctflow/lvfm/cluster/lvfm_UNetSTIC-S_16f8_all.yaml

accelerate launch --num_processes 3 --multi_gpu --num_machines 1 --mixed_precision bf16 ctflow/lvfm/train.py --config ctflow/lvfm/configs/jiayi_lvdm_STDiT-S2_16f8_all.yaml
"""


allow_ops_in_compiled_graph()

torch._dynamo.config.suppress_errors = True

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker")

logger = get_logger(__name__, log_level="INFO")

SCALE = 0.8487  # latent scale for finetuned VAE (std of raw latents)


def _load_text_encoder(text_encoder_path, tokenizer_name):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(tokenizer_name, trust_remote_code=True)
    state_dict = torch.load(text_encoder_path, map_location="cpu")
    text_sd = {k.replace("text_transformer.", ""): v
               for k, v in state_dict.items() if k.startswith("text_transformer.")}
    model.load_state_dict(text_sd, strict=False)
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model.eval()


def _encode_text_batch(texts, tokenizer, text_encoder, device, dtype, max_length=512):
    from transformers import AutoTokenizer
    inputs = tokenizer(
        list(texts), return_tensors="pt", truncation=True,
        padding="max_length", max_length=max_length
    )
    with torch.no_grad():
        out = text_encoder(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    emb = out.last_hidden_state[:, 0:1, :]          # CLS: [B, 1, 768]
    emb = F.normalize(emb.float(), p=2, dim=-1)
    return emb.to(dtype)


# ARGS
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, help="Path to the config file.")
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable wandb logging.", default=False
    )
    args = parser.parse_args()
    return args

def log_validation(
    config, maybe_ema_denoiser, accelerator, weight_dtype, val_dataset, step=None,
    text_encoder=None, tokenizer=None,
):
    logger.info("Running validation... ")

    val_vae = instantiate(config.vae)
    ft_vae_ckpt = os.path.join(config.vae.pretrained, "vae_step11000.pt")
    val_vae.load_state_dict(torch.load(ft_vae_ckpt, map_location="cpu", weights_only=False))
    val_vae = val_vae.eval().to(accelerator.device)

    val_denoiser = instantiate_class_from_config(config.denoiser).eval()
    if not ("UNetSTIC" in config.denoiser.target):
        val_denoiser.enable_xformers_memory_efficient_attention()

    if config.get("use_ema", False):
        maybe_ema_denoiser.copy_to(val_denoiser.parameters())
    else:
        with torch.no_grad():
            for param, eval_param in zip(
                maybe_ema_denoiser.parameters(), val_denoiser.parameters()
            ):
                eval_param.copy_(param)
    val_denoiser.to(accelerator.device, weight_dtype)

    if config.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    indices = torch.randint(0, len(val_dataset), (config.validation.samples,), device=accelerator.device).tolist()

    B, C, T, H, W = (
        config.validation.samples,
        config.globals.latent_channels,
        config.validation.frames,
        config.globals.latent_res,
        config.globals.latent_res,
    )

    ref_elements = [val_dataset[i] for i in indices]

    ref_images = [e["image"] for e in ref_elements]
    ref_images = torch.stack(ref_images, dim=0)  
    ref_images = ref_images.to(accelerator.device, dtype=weight_dtype)
    ref_images = sample_latents(config, ref_images)
    ref_images = ref_images * SCALE  # B x C x T x H x W

    ref_videos = [e["video"] for e in ref_elements]  # B - C x T x H x W
    ref_videos = torch.stack(ref_videos, dim=0)
    ref_videos = ref_videos.to(device=accelerator.device)  # B x C x T x H x W
    ref_videos = sample_latents(config, ref_videos)

    if text_encoder is not None and "text" in ref_elements[0]:
        texts = [e["text"] for e in ref_elements]
        text_embeddings = _encode_text_batch(
            texts, tokenizer, text_encoder, accelerator.device, weight_dtype,
            max_length=config.get("text_max_length", 512),
        )
    else:
        text_embeddings = [e["embedding"] for e in ref_elements]
        text_embeddings = torch.stack(text_embeddings, dim=0)
        text_embeddings = text_embeddings.to(accelerator.device, dtype=weight_dtype)

    logger.info("Sampling... ")
    with torch.no_grad(), accelerator.autocast():
        # prepare model inputs
        z_1 = torch.randn(
            (B, C, T, H, W),
            device=accelerator.device,
            dtype=weight_dtype,
            generator=generator,
        )

        #lvefs = torch.tensor(
        #    config.validation.lvefs, device=accelerator.device, dtype=weight_dtype
        #)
        #lvefs = lvefs[:, None, None]  # B -> B x 1 x 1
        ## dummy lvef score ###
        #fake_lvef = torch.full((B,), -1.0, device=accelerator.device)         # [B]
        #fake_lvef_expanded = fake_lvef[:, None, None]  # [B, 1, 1]
        # z_1 = torch.cat([z_1, ref_images], dim=1)  # B x 2C x T x H x W

        timesteps = torch.tensor(
            [1.0, 0.0], dtype=weight_dtype, device=accelerator.device
        )

        # path val_denoiser with CFG
        val_denoiser.forward_original = val_denoiser.forward
        cfg_scale = config.get("guidance_scale", 1.0)
        null_text = torch.zeros_like(text_embeddings)
        null_image = torch.zeros_like(ref_images)

        def new_forward(self, t, y, *args, **kwargs):
            v_cond = self.forward_original(y, t, encoder_hidden_states=text_embeddings, cond_image=ref_images).sample
            v_uncond = self.forward_original(y, t, encoder_hidden_states=null_text, cond_image=null_image).sample
            return v_uncond + cfg_scale * (v_cond - v_uncond)

        val_denoiser.forward = types.MethodType(new_forward, val_denoiser)

        synthetic_video = odeint(
            val_denoiser,
            z_1,
            timesteps,
            atol=1e-5,
            rtol=1e-5,
            adjoint_params=val_denoiser.parameters(),
            # method=config.validation.method,
        )[-1]

    # VAE decoding
    with torch.no_grad():  # no autocast
        synthetic_video = rearrange(synthetic_video, "b c t h w -> (b t) c h w")
        synthetic_video = synthetic_video / SCALE
        synthetic_video = val_vae.decode(synthetic_video.float()).sample
        synthetic_video = synthetic_video.clamp(-1, 1).add(1).div(2).mul(255)  # [-1, 1] -> [0, 255]
        synthetic_video = synthetic_video.clamp(0, 255).to(torch.uint8).cpu()
        synthetic_video = rearrange(synthetic_video, "(b t) c h w -> b c t h w", b=B)

        ref_images = rearrange(ref_images, "b c t h w -> (b t) c h w")
        ref_images = ref_images / SCALE
        ref_images = val_vae.decode(ref_images.float()).sample
        ref_images = ref_images.clamp(-1, 1).add(1).div(2).mul(255)  # [-1, 1] -> [0, 255]
        ref_images = ref_images.clamp(0, 255).to(torch.uint8).cpu()
        ref_images = rearrange(ref_images, "(b t) c h w -> b c t h w", b=B)

        ref_videos = rearrange(ref_videos, "b c t h w -> (b t) c h w")
        ref_videos = val_vae.decode(ref_videos.float()).sample
        ref_videos = ref_videos.clamp(-1, 1).add(1).div(2).mul(255)  # [-1, 1] -> [0, 255]
        ref_videos = ref_videos.clamp(0, 255).to(torch.uint8).cpu()
        ref_videos = rearrange(ref_videos, "(b t) c h w -> b c t h w", b=B)

        videos = torch.cat(
            [ref_images, ref_videos, synthetic_video], dim=3
        )  # B x C x T x 3H x W

    # reshape for wandb
    videos = rearrange(videos, "b c t h w -> t c h (b w)")  # prepare for wandb
    if step is not None:
        fname = f"sample_{step:07d}.mp4"
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        save_as_mp4(
            rearrange(videos, "t c h w -> t h w c"),
            os.path.join(config.output_dir, "samples", fname),
        )
    videos = videos.numpy()

    logger.info("Done sampling... ")
    if config.validation.fps == "original":
        config.validation_fps = 50.0
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": wandb.Video(
                        videos,
                        caption=(
                            "Lvefs: -1"
                        ),
                        fps=config.validation.fps,
                    )
                }
            )
            logger.info("Samples sent to wandb.")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    # val_denoiser.forward = val_denoiser.forward_original
    # del val_denoiser.forward_original

    del val_denoiser
    del val_vae
    torch.cuda.empty_cache()

    return videos

# MAIN
def main():

    print("Code starting...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    # torch.autograd.set_detect_anomaly(True)  # If NaNs are detected, run with this flag

    args = parse_args()
    config = OmegaConf.load(args.config)
    
    logger = get_logger(__name__, log_level="INFO")
    accelerator, logger = setup_accelerator_and_logging(config, logger)
    print(f"Accelerator ready for process {accelerator.process_index}")

    # MODELS, OPTIMIZER, DATA
    denoiser = instantiate_class_from_config(config.denoiser).train()
    if not ("UNetSTIC" in config.denoiser.target):
        denoiser.enable_xformers_memory_efficient_attention()
    clamp_v = config.get("clamp_v", float("inf"))

    ema_denoiser = get_ema_model(denoiser) if config.use_ema else None
    optimizer = instantiate_class_from_config(config.optimizer, denoiser.parameters())
    #lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)

    # train_dataset = instantiate_cached_dataset_sm(
    #     config.datasets,
    #     split=["TRAIN"],
    #     is_main_process=accelerator.is_local_main_process,
    # )

    #train_dataset = instantiate_cached_dataset(config.datasets, split=["TRAIN"])

    #val_dataset = instantiate_dataset(config.datasets, split=["VAL"])
    train_dataset = instantiate_dataset(config.datasets, split=["TRAIN"])
    val_dataset = instantiate_dataset(config.val_datasets, split=["VAL"])

    # on-the-fly text encoder (optional — activated by text_encoder_path in config)
    _tokenizer, _text_encoder = None, None
    if config.get("text_encoder_path"):
        _tokenizer, _text_encoder = _load_text_encoder(
            config.text_encoder_path, config.tokenizer_name
        )
        _text_encoder = _text_encoder.to(accelerator.device)
        logger.info(f"Text encoder loaded from {config.text_encoder_path}")
    #lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)
    #rng = np.random.default_rng(42)
    train_dataloader = instantiate_class_from_config(
        config.dataloader,
        train_dataset,
        #collate_fn=custom_collate_fn,
    )

    # Wrap for on-device / distributed training
    denoiser, optimizer, train_dataloader = accelerator.prepare(
        denoiser, optimizer, train_dataloader
    )
    lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)


    # ENV SETUP
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )
    dtype, config = set_weight_dtype(accelerator, config)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # RESUME
    """
    initial_global_step, first_epoch = load_checkpoint(
        config, accelerator, num_update_steps_per_epoch, ema_denoiser
    )
    if config.get("use_ema", False):
        ema_denoiser.to(accelerator.device)
    global_step = initial_global_step
    if global_step == 0:
        set_seed(config.seed)
    """
    # RESUME / FINETUNE
    initial_global_step, first_epoch = load_checkpoint(
        config, accelerator, num_update_steps_per_epoch, ema_denoiser
    )
    if config.get("use_ema", False):
        ema_denoiser.to(accelerator.device)

    if_ft = bool(config.get("if_fine_tuned", False))

    if if_ft:
        # —— fine tuned：keep weight，but from step 0 —— 
        global_step = 0

        # use old config
        optimizer = instantiate_class_from_config(config.optimizer, denoiser.parameters())
        lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)

    else:
        global_step = initial_global_step

    if global_step == 0:
        set_seed(config.seed)

    # INITIALIZE WANDB
    init_trackers(accelerator, config, args)
    # if accelerator.is_main_process:
    #     wandb.watch(accelerator.unwrap_model(denoiser), log_freq=100, log="all")
    log_training_info(config, accelerator, denoiser, train_dataset, logger)

    forward_kwargs = prepare_forward_kwargs(config, denoiser, accelerator)
    if config.max_grad_norm < 0:
        config.max_grad_norm = float("inf")

    # LOADING BAR AND INFO
    infinite_loader = cycle(train_dataloader)  # needs to be called after prepare
    print(type(train_dataloader))
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    # START TRAINING
    while True:
        batch = next(infinite_loader)
        with accelerator.accumulate(denoiser):

            images = sample_latents(config, batch["image"]).detach()  # BCHW
            images = images * SCALE
            images = images.clamp(-clamp_v, clamp_v)

            videos = sample_latents(config, batch["video"]).detach()  # BCTHW
            videos = videos * SCALE
            videos = videos.clamp(-clamp_v, clamp_v)

            B, C, T, H, W = videos.shape

            """
            lvefs = batch["lvef"]  # B
            padding_indices = batch["padding"]  # B

            B, C, T, H, W = videos.shape

            if config.get("no_conditionning", False):
                images *= 0.0  # set all images to 0.0
                lvefs = lvefs * 0.0 - 1.0  # set all lvefs to -1.0

            # LVEFS, -1 means no conditionning
            lvefs_mask = (torch.rand_like(lvefs) > config.p_drop_conditionning).float()
            lvefs = lvefs * lvefs_mask + (1 - lvefs_mask) * -1.0
            lvefs = lvefs[:, None, None]  # B -> B x 1 x 1
            forward_kwargs["encoder_hidden_states"] = lvefs
            """
            ## new lvef score ###
            #fake_lvef = torch.full((B,), -1.0)         # [B]
            #fake_lvef_expanded = fake_lvef[:, None, None]  # [B, 1, 1]
            #forward_kwargs["encoder_hidden_states"] = fake_lvef_expanded
            if _text_encoder is not None and "text" in batch:
                text_embeddings = _encode_text_batch(
                    batch["text"], _tokenizer, _text_encoder,
                    accelerator.device, dtype,
                    max_length=config.get("text_max_length", 512),
                )
            else:
                text_embeddings = batch["embedding"]
            # CFG dropout for text embedding
            text_mask = (torch.rand(B, device=accelerator.device) > config.p_drop_conditionning).float()
            text_mask = text_mask[:, None, None]  # [B, 1, 1]
            text_embeddings = text_embeddings * text_mask
            forward_kwargs["encoder_hidden_states"] = text_embeddings

            # IMAGES
            """
            if config.get("noise_cond_image", 0.0) > 0.0:
                images = images + config.noise_cond_image * torch.randn_like(
                    images[:, :, :, :]  # same noise on all frames
                )

            images_mask = (
                torch.rand_like(images[:, 0:1, 0:1, 0:1]) > config.p_drop_conditionning
            ).float()  # B x 1 x 1 x 1
            images = (
                images * images_mask + (1.0 - images_mask) * 0.0
            )  # 0 means no conditionning

            if config.get("single_cond_image", False):
                images_zero = torch.zeros(
                    (B, C, T, H, W), device=accelerator.device, dtype=dtype
                )
                images_zero[:, :, 0, :, :] = images[:, :, :, :]  # B x C x T x H x W
                images = images_zero
            else:
                images = images[:, :, None, :, :]  # B x C x 1 x H x W
                images = images.repeat(1, 1, T, 1, 1)  # B x C x T x H x W

            forward_kwargs["cond_image"] = images
            """
            # whole current block as condition image
            if config.get("noise_cond_image", 0.0) > 0.0:
                images += config.noise_cond_image * torch.randn_like(images)

            images_mask = (
                torch.rand_like(images[:, 0:1, 0:1, 0:1]) > config.p_drop_conditionning
            ).float()  # [B, 1, 1, 1]
            images = images * images_mask + (1.0 - images_mask) * 0.0
            forward_kwargs["cond_image"] = images

            # TIMESTEPS for the noise scheduler
            t = torch.rand(B, device=accelerator.device, dtype=dtype)
            t = t.view(-1, 1, 1, 1, 1)  # B x 1 x 1 x 1 x 1
            # squeeze but keep the batch dimension
            forward_kwargs["timestep"] = t[:, 0, 0, 0, 0]

            # NOISE the latents
            z_0 = videos  # B x C x T x H x W
            z_1 = get_noise(videos, noise_offset=config.get("noise_offset", 0.0))
            offset = 1e-5
            z_t = (1 - t) * z_0 + (offset + (1 - offset) * t) * z_1  # interpolation
            u = (1 - offset) * z_1 - z_0  # velocity

            with accelerator.autocast():
                # Forward pass
                v = denoiser(z_t, **forward_kwargs).sample

                # Compute loss
                loss = F.mse_loss(v, u)

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    denoiser.parameters(), config.max_grad_norm
                )
                if config.max_grad_value > 0:
                    accelerator.clip_grad_value_(
                        denoiser.parameters(), config.max_grad_value
                    )
                lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # LOADING BAR
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # EMA and logging
            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_denoiser.step(denoiser.parameters())
                progress_bar.update(1)
            if accelerator.sync_gradients and global_step % 10 == 0:
                accelerator.log(
                    {
                        "train_loss": loss.mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(),
                        "mean_latent": videos.mean().item(),
                        "std_latent": videos.std().item(),
                    },
                    step=global_step,
                )

            # VALIDATION
            if (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and global_step % config.validation.steps == 0
            ):
                # if config.use_ema:
                # ema_denoiser.store(denoiser.parameters())
                # ema_denoiser.copy_to(denoiser.parameters())

                log_validation(
                    config,
                    ema_denoiser or denoiser,
                    accelerator,
                    dtype,
                    val_dataset,
                    step=global_step,
                    text_encoder=_text_encoder,
                    tokenizer=_tokenizer,
                )

                # if config.use_ema:
                #     ema_denoiser.restore(denoiser.parameters())

            # CHECKPOINTING
            if (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and global_step % config.checkpointing_steps == 0
            ):
                cleanup_checkpoints(config, logger)
                save_checkpoint(config, accelerator, logger, global_step, ema_denoiser)

            # STOPPING CONDITION
            if global_step >= config.max_train_steps:
                break

            if accelerator.sync_gradients:
                global_step += 1

    cleanup_checkpoints(config, logger)  # cleanup the last unnecessary checkpoint
    accelerator.end_training()


if __name__ == "__main__":
    main()
