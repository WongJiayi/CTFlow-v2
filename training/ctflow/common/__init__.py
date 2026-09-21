# Standard library imports
import argparse
import importlib
import inspect
import json
import logging
import math
import os
import shutil
from enum import Enum
from functools import partial

# Third-party library imports
import cv2
# Specific imports from diffusers
import diffusers
import imageio
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
# Specific imports from accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from safetensors.torch import load_file, save_file
from sklearn.neighbors import KNeighborsClassifier
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import tqdm as tqdm_std

import wandb

# tqdm = partial(tqdm_std, dynamic_ncols=True, mininterval=1.0)


class Scheduler(Enum):
    EDM = 0
    EULER = 1
    OTHER = 2


logger = get_logger(__name__, log_level="INFO")

### Instantiation helper functions ###


def parse_klass_arg(value, full_config):
    """
    Parse an argument value that might represent a class, enum, or basic data type.
    This function tries to dynamically import and resolve nested attributes.
    It also resolves OmegaConf interpolations if found.
    """
    if isinstance(value, str) and "." in value:
        # Check if the value is an interpolation and try to resolve it
        if value.startswith("${") and value.endswith("}"):
            try:
                # Attempt to resolve the interpolation directly using OmegaConf
                value = omegaconf.OmegaConf.resolve(full_config)[value[2:-1]]
            except Exception as e:
                logger.error(f"Error resolving OmegaConf interpolation {value}: {e}")
                return None

        parts = value.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            attr_name = parts[i]
            try:
                module = importlib.import_module(module_name)
                result = module
                for j in range(i, len(parts)):
                    result = getattr(result, parts[j])
                return result
            except ImportError as e:
                continue
            except AttributeError as e:
                logger.warning(
                    f"Warning: Could not resolve attribute {parts[j]} from {module_name}, error: {e}"
                )
                continue
        # print(f"Warning: Failed to import or resolve {value}. Falling back to string.")
        return (
            value  # Return the original string if no valid import and resolution occurs
        )
    return value


def instantiate_class_from_config(config, *args, **kwargs):
    """
    Dynamically instantiate a class based on a configuration object.
    Supports passing additional positional and keyword arguments.
    """
    module_name, class_name = config.target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    # Assuming config might be a part of a larger OmegaConf structure:
    # if not isinstance(config, omegaconf.DictConfig):
    #     config = omegaconf.OmegaConf.create(config)
    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    # Resolve args and kwargs from the configuration
    # conf_args = [parse_klass_arg(arg, config) for arg in config.get('args', [])]
    # conf_kwargs = {key: parse_klass_arg(value, config) for key, value in config.get('kwargs', {}).items()}
    conf_kwargs = {
        key: parse_klass_arg(value, config) for key, value in config["args"].items()
    }
    # Combine conf_args with explicitly passed *args
    all_args = list(args)  # + conf_args

    # Combine conf_kwargs with explicitly passed **kwargs
    all_kwargs = {**conf_kwargs, **kwargs}

    # Instantiate the class with the processed arguments
    instance = klass(*all_args, **all_kwargs)
    return instance


### Accelerator and logging setup ###


def setup_accelerator_and_logging(config, logger):
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,  # fix the bug with the scheduler
        # kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        # rng_types=["python", "torch"],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    return accelerator, logger


def set_weight_dtype(accelerator, config):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision
    return weight_dtype, config


def init_trackers(accelerator, config, args):
    if accelerator.is_main_process:
        tracker_config = omegaconf.OmegaConf.to_container(config, resolve=True)
        wandb_args = omegaconf.OmegaConf.to_container(config.wandb_args, resolve=True)
        accelerator.init_trackers(
            project_name=wandb_args.pop("project"),
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    **wandb_args,
                    "mode": "disabled" if args.no_wandb else "online",
                    "resume": "allow",
                }
            },
        )
        config.wandb_args.id = wandb.run.id


def log_training_info(config, accelerator, model, train_dataset, logger):
    total_batch_size = (
        config.dataloader.args.batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    model_num_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset):_}")
    logger.info(f"  Num Epochs = {config.num_train_epochs:_}")
    logger.info(
        f"  Instantaneous batch size per device = {config.dataloader.args.batch_size:_}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size:_}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.gradient_accumulation_steps:_}"
    )
    logger.info(f"  Total optimization steps = {config.max_train_steps:_}")
    logger.info(
        f"  Model: Total params = {model_num_params:_} \t Trainable params = {model_trainable_params:_} ({model_trainable_params/model_num_params*100:.2f}%)"
    )

    if config.get("scheduler_type", None) is not None:
        logger.info(f"  Scheduler type: {config.scheduler_type.name}")


### Checkpointing and model saving/loading ###


def load_checkpoint(config, accelerator, num_update_steps_per_epoch, ema_model=None):
    first_epoch = 0
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            if ema_model is not None:
                ema_path = os.path.join(config.output_dir, path, "denoiser_ema")
                model_cls = accelerator._models[0].__class__
                ema_tmp = ema_model.__class__.from_pretrained(
                    ema_path, model_cls=model_cls
                )
                # Replace the existing ema_model with the loaded one
                ema_model.__dict__.update(ema_tmp.__dict__)

            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            # get the wandb unique id
            run_config = omegaconf.OmegaConf.load(
                os.path.join(config.output_dir, "config.yaml")
            )
            config.wandb_args.id = run_config.wandb_args.id
    else:
        initial_global_step = 0

    return initial_global_step, first_epoch


def cleanup_checkpoints_with_limit(config, logger):
    if config.checkpoints_total_limit is not None:
        checkpoints = os.listdir(config.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= config.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(
                    config.output_dir, removing_checkpoint
                )
                shutil.rmtree(removing_checkpoint)


def cleanup_checkpoints(config, logger):
    if config.get("checkpoints_total_limit", None) is not None:
        cleanup_checkpoints_with_limit(config, logger)

    elif config.get("checkpoints_to_keep", None) is not None:
        checkpoints_to_keep = omegaconf.OmegaConf.to_container(
            config.checkpoints_to_keep
        )
        checkpoints_to_keep = [f"checkpoint-{c}" for c in checkpoints_to_keep]

        checkpoints = os.listdir(config.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) == 0:
            return

        last_checkpoint = checkpoints[-1]

        for checkpoint in checkpoints:
            if checkpoint not in checkpoints_to_keep and checkpoint != last_checkpoint:
                logger.info(f"Removing checkpoint: {checkpoint}")
                checkpoint_path = os.path.join(config.output_dir, checkpoint)
                shutil.rmtree(checkpoint_path)


def save_checkpoint(config, accelerator, logger, global_step, ema_model=None):
    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if ema_model is not None:
        ema_model.save_pretrained(os.path.join(save_path, "denoiser_ema"))
    omegaconf.OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    logger.info(f"Saved state to {save_path}")


def save_model_hook(models, weights, output_dir):

    for i, model in enumerate(models):
        has_saved = False
        for nn_name in ["net", "encoder", "backbone"]:
            if hasattr(model, nn_name):
                state_dict = getattr(model, nn_name).state_dict()
                save_file(
                    state_dict, os.path.join(output_dir, f"{nn_name}.safetensors")
                )
                has_saved = True
                break

        if not has_saved:
            model.save_pretrained(os.path.join(output_dir, "denoiser"))

        weights.pop()


def load_model_hook(models, input_dir):

    for i in range(len(models)):
        model = models.pop()
        has_loaded = False
        for nn_name in ["net", "encoder", "backbone"]:
            if hasattr(model, nn_name):
                state_dict = load_file(
                    os.path.join(input_dir, f"{nn_name}.safetensors")
                )
                getattr(model, nn_name).load_state_dict(state_dict)
                has_loaded = True
                break

        if not has_loaded:
            # load diffusers style into model
            load_model = model.__class__.from_pretrained(
                os.path.join(input_dir, "denoiser")
            )
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

            model.from_pretrained(os.path.join(input_dir, "denoiser"))



def load_weights_from_config(model, config, file_name="backbone.safetensors") -> None:
    """
    Load weights from a checkpoint file specified in the config.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        config (omegaconf.DictConfig): The configuration object.
        file_name (str, optional): The name of the file to load the weights from. Defaults to "backbone.safetensors".

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """

    if not os.path.exists(config.output_dir):
        raise FileNotFoundError(f"Could not find output directory {config.output_dir}")

    checkpoints = os.listdir(config.output_dir)
    last_checkpoint = sorted([c for c in checkpoints if c.startswith("checkpoint")])[-1]
    checkpoint_path = os.path.join(config.output_dir, last_checkpoint, file_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint file {checkpoint_path}")

    weights = load_file(checkpoint_path)

    model.load_state_dict(weights)

    return model


### Data loading helper ###


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def resamp_collate_fn(batch, dataset):
    resampled = []
    for data in batch:
        while data["images"] is None:  # Check if the image is None
            idx = torch.randint(0, len(dataset), (1,)).item()
            data = dataset.__getitem__(idx)  # Randomly resample from dataset
        resampled.append(data)

    output_dict = {}
    for key in resampled[0].keys():
        if key == "images":
            output_dict[key] = torch.stack([item[key] for item in resampled])
        else:
            output_dict[key] = torch.tensor([item[key] for item in resampled])
    return output_dict


### Visualization helper functions ###


def create_color_mask(single_channel_mask, color):
    """
    Convert a single-channel mask to a color mask.

    Parameters:
    - single_channel_mask: numpy array, single-channel mask
    - color: tuple of 3 integers, the color to apply (in BGR format)

    Returns:
    - color_mask: numpy array, color mask with the same dimensions as the input mask
    """
    # Create an empty color mask with the same dimensions as the input mask but with 3 channels
    color_mask = np.zeros(
        (single_channel_mask.shape[0], single_channel_mask.shape[1], 3), dtype=np.uint8
    )

    # Apply the specified color to the regions where the mask is greater than zero
    color_mask[single_channel_mask > 0] = color

    return color_mask


def binarize_mask(mask: torch.Tensor, threshold=0.5):
    """
    Binarizes a mask tensor using a threshold.

    Args:
        mask (torch.Tensor): The mask tensor to binarize.
        threshold (float, optional): The threshold to use for binarization. Defaults to 0.5.

    Returns:
        torch.Tensor: The binarized mask tensor.
    """
    return mask.sigmoid().ge(threshold).float()


### Other helper functions ###


def padf(tensor, res=16, mode="circular"):
    """
    Pads a tensor along its last two dimensions to make their sizes multiples of 'res'.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        res (int, optional): The resolution to pad to. Defaults to 16.
        mode (str, optional): Padding mode. Defaults to 'circular'.

    Returns:
        torch.Tensor: The padded tensor.
        int: The amount of padding applied.
    """
    pad = (
        res - tensor.shape[-1] % res
    ) % res  # Handle case when dimension is already a multiple
    pad = pad // 2
    padding = [pad, pad, pad, pad]  # Padding for last two dimensions
    tensor = F.pad(tensor, (pad, pad, pad, pad, 0, 0), mode=mode)
    return tensor, pad


def unpadf(tensor, pad=1):
    """
    Removes padding from a tensor along the last two dimensions.

    Args:
        tensor (torch.Tensor): The tensor to unpad.
        pad (int, optional): The amount of padding to remove. Defaults to 1.

    Returns:
        torch.Tensor: The unpadded tensor.
    """
    return tensor[..., pad:-pad, pad:-pad]


def pad_reshape(tensor, mult=3):
    """
    Pads a tensor along the last dimension to make its size a multiple of 2^mult and reshapes it.

    Args:
        tensor (torch.Tensor): The tensor to pad and reshape.
        mult (int, optional): The power of 2 that the tensor's size should be a multiple of. Defaults to 3.

    Returns:
        torch.Tensor: The padded and reshaped tensor.
        int: The amount of padding applied.
    """
    tensor, pad = padf(tensor, res=2**mult)
    tensor = rearrange(tensor, "b c t h w -> b t c h w")
    return tensor, pad


def unpad_reshape(tensor, pad=1):
    """
    Reshapes a tensor and removes padding from it along the last two dimensions.

    Args:
        tensor (torch.Tensor): The tensor to reshape and unpad.
        pad (int, optional): The amount of padding to remove. Defaults to 1.

    Returns:
        torch.Tensor: The reshaped and unpadded tensor.
    """
    tensor = rearrange(tensor, "b t c h w -> b c t h w")
    tensor = unpadf(tensor, pad=pad)
    return tensor


def instantiate_from_config(
    config, scope: list[str], return_klass_kwargs=False, **kwargs
):
    """
    Instantiate a class from a config dictionary.

    Args:
        config (dict): The config dictionary.
        scope (list[str]): The scope of the class to instantiate.
        return_klass_kwargs (bool, optional): Whether to return the class and its kwargs. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the class constructor.

    Returns:
        object: The instantiated class.
        (optional) type: The class that was instantiated.
        (optional) dict: The kwargs that were passed to the class constructor.
    """
    okwargs = omegaconf.OmegaConf.to_container(config, resolve=True)
    klass_name = okwargs.pop("_class_name")
    klass = None

    for module_name in scope:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue  # Try next module

        klass = getattr(module, klass_name, None)
        if klass is not None:
            break  # Stop when we find a matching class

    assert (
        klass is not None
    ), f"Could not find class {klass_name} in the specified scope"
    instance = klass(**okwargs, **kwargs)

    if return_klass_kwargs:
        return instance, klass, okwargs
    return instance


def filter_kwargs_for_func(func, kwargs):
    init_signature = inspect.signature(func)
    # Extract the names of its parameters, skipping 'self'
    valid_keys = init_signature.parameters.keys()
    # Filter the kwargs to only include valid keys
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    return filtered_kwargs


def instantiate(config_og, return_klass_kwargs=False):
    config = omegaconf.OmegaConf.create(config_og)  # Make sure we have a copy
    #logger.info(f"Loading {config.target.split('.')[-1]} from config...")
    module_path, class_name = config.target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if "pretrained" in config:
        pretrained = config.pop("pretrained")
        fkwargs = {"use_safetensor": True, "torch_dtype": torch.float32}
        fkwargs = filter_kwargs_for_func(cls.from_pretrained, fkwargs)
        if "," in pretrained:
            pretrained, subfolder = pretrained.split(",")
            obj = cls.from_pretrained(pretrained, subfolder=subfolder, **fkwargs)
        else:
            obj = cls.from_pretrained(pretrained, **fkwargs)
        if return_klass_kwargs:
            config["args"] = (
                filter_kwargs_for_func(cls.__init__, dict(obj.config))
                if hasattr(obj, "config")
                else {}
            )
    elif "weights" in config:
        weights_path = config.pop("weights")
        weigths = torch.load(weights_path)
        obj = cls(**omegaconf.OmegaConf.to_container(config.args))
        obj.load_state_dict(weights)
    else:
        obj = cls(**omegaconf.OmegaConf.to_container(config.args))

    if return_klass_kwargs:
        return obj, cls, omegaconf.OmegaConf.to_container(config.args)
    else:
        return obj


def load_model(path):
    """
    Loads a model from a checkpoint.

    Args:
        path (str): The path to the checkpoint.

    Returns:
        object: The loaded model.
    """
    # find config.json
    json_path = os.path.join(path, "config.json")
    assert os.path.exists(json_path), f"Could not find config.json at {json_path}"
    with open(json_path, "r") as f:
        config = json.load(f)

    # instantiate class
    klass_name = config["_class_name"]
    klass = getattr(diffusers, klass_name, None)
    if klass is None:
        klass = globals().get(klass_name, None)
    assert (
        klass is not None
    ), f"Could not find class {klass_name} in diffusers or global scope."
    assert (
        getattr(klass, "from_pretrained", None) is not None
    ), f"Class {klass_name} does not support 'from_pretrained'."

    # load checkpoint
    model = klass.from_pretrained(path)

    return model


def save_as_mp4(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as an MP4 video.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the video.
    - fps: int. Frames per second for the output video.

    Returns:
    - None
    """
    import imageio

    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()

    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")

    # Write the frames to a video file
    with imageio.get_writer(
        filename,
        fps=fps,
    ) as writer:
        for i in range(np_video.shape[0]):
            writer.append_data(np_video[i])


def save_as_avi(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as an AVI video with reduced compression.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the video.
    - fps: int. Frames per second for the output video.

    Returns:
    - None
    """
    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()

    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")

    # Define codec for reduced compression
    codec = "mjpeg"  # MJPEG codec for AVI files
    # High quality (lower values mean higher quality, but larger file sizes)
    quality = 10
    # pixel_format = "yuvj420p"
    # Write the frames to a video file
    with imageio.get_writer(
        filename,
        fps=fps,
        codec=codec,
        quality=quality,
    ) as writer:
        for frame in np_video:
            writer.append_data(frame)


def save_as_gif(tensor, filename, fps=30):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as a GIF.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the GIF.
    - fps: int. Frames per second for the output GIF.

    Returns:
    - None
    """
    import imageio

    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()

    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")

    # Write the frames to a GIF file
    imageio.mimsave(filename, np_video, fps=fps, loop=0)


def save_as_img(tensor, filename, ext="jpg"):
    """
    Saves a 4D tensor (nFrames, height, width, channels) as a series of JPG images.
    OR
    Saves a 3D tensor (height, width, channels) as a single image.

    Parameters:
    - tensor: 4D torch.Tensor. Tensor containing the video frames.
    - filename: str. The output filename for the JPG images.

    Returns:
    - None
    """
    import imageio

    # Make sure the tensor is on the CPU and is a numpy array
    np_video = tensor.cpu().numpy()

    # Ensure the tensor dtype is uint8
    if np_video.dtype != np.uint8:
        raise ValueError("The tensor has to be of type uint8")

    # Write the frames to a series of JPG files
    if len(np_video.shape) == 3:
        imageio.imwrite(filename, np_video, quality=100)
    else:
        os.makedirs(filename, exist_ok=True)
        for i in range(np_video.shape[0]):
            imageio.imwrite(
                os.path.join(filename, f"{i:04d}.{ext}"), np_video[i], quality=100
            )


def loadvideo(filename: str, return_fps=False):
    """
    Loads a video file into a tensor of frames.

    Args:
        filename (str): The path to the video file.
        return_fps (bool, optional): Whether to return the frames per second of the video. Defaults to False.

    Raises:
        FileNotFoundError: If the video file does not exist.

    Returns:
        torch.Tensor: A tensor of the video's frames, with shape (frames, 3, height, width).
        (optional) float: The frames per second of the video. Only returned if return_fps is True.
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)  # type: ignore

    fps = capture.get(cv2.CAP_PROP_FPS)  # type: ignore

    frames = []

    while True:  # load all frames
        ret, frame = capture.read()
        if not ret:
            break  # Reached end of video
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame)

        frames.append(frame)
    capture.release()

    frames = torch.stack(frames, dim=0)  # (frames, 3, height, width)

    if return_fps:
        return frames, fps
    return frames


def parse_formats(s):
    # Split the input string by comma and strip spaces
    formats = [format.strip().lower() for format in s.split(",")]
    # Define the allowed choices
    allowed_formats = ["avi", "mp4", "gif", "jpg", "png", "pt"]
    # Check if all elements in formats are in allowed_formats
    for format in formats:
        if format not in allowed_formats:
            raise argparse.ArgumentTypeError(
                f"{format} is not a valid format. Choose from {', '.join(allowed_formats)}."
            )
    return formats


def get_dtype(config):
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtypes.get(config.mixed_precision, torch.float32)


def is_segmentation_conditionned(config):
    # detect if the model is segmentation-conditionned
    dataset_has_segmentation = (
        config.datasets[0].params.get("segmentation_root", None) != None
    )
    denoiser_has_segmentation = (
        config.denoiser.args.in_channels > config.globals.latent_channels
    )
    segmentation_conditionned = None
    if dataset_has_segmentation != denoiser_has_segmentation:
        raise ValueError(
            f"One of the dataset ({dataset_has_segmentation}) or the denoiser ({denoiser_has_segmentation}) is segmentation-conditionned, but not both."
        )
    elif dataset_has_segmentation:
        segmentation_conditionned = True
        logger.info("Segmentation-conditionned model detected.")
    else:
        segmentation_conditionned = False

    return segmentation_conditionned


def is_class_conditionned(config):
    if "view" in config.globals.outputs:
        logger.info("View-conditionned model detected.")
    return "view" in config.globals.outputs


def class_condition(batch, forward_kwargs, accelerator):
    class_labels = batch["view"]
    class_labels = torch.tensor(class_labels, device=accelerator.device).long()
    forward_kwargs["class_labels"] = class_labels
    return forward_kwargs


def prepare_forward_kwargs(config, denoiser, accelerator):
    forward_kwargs = {
        "timestep": None,
    }

    # check if denoiser.forward has a class_labels argument
    if "class_labels" in inspect.signature(denoiser.forward).parameters:
        forward_kwargs["class_labels"] = torch.zeros(
            (config.dataloader.args.batch_size,), device=accelerator.device
        ).long()
    if "encoder_hidden_states" in inspect.signature(denoiser.forward).parameters:
        forward_kwargs["encoder_hidden_states"] = torch.zeros(
            (
                config.dataloader.args.batch_size,
                1,
                config.denoiser.args.get("joint_attention_dim", 1),
            ),
            device=accelerator.device,
        )

    return forward_kwargs


def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    else:
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))


def initialize_weights(module):
    """
    Initialize the weights of the model's layers.
    """
    if isinstance(module, nn.Conv2d):
        # Initialize Conv2d weights with Kaiming Normal (He initialization)
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        # Initialize Linear weights with Xavier Uniform initialization
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.GroupNorm):
        # Initialize GroupNorm weights and biases
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm weights and biases
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        # Initialize Embedding weights
        nn.init.normal_(module.weight, mean=0, std=0.02)

    elif isinstance(module, nn.MultiheadAttention):
        # Initialize MultiheadAttention weights
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def zero_nan_inf_all_grad(model):
    """
    Replace NaN and Inf values in all parameter gradients of a model without looping
    over parameters at Python level. Instead, we:
    1. Flatten all gradients into a single vector.
    2. Clean the vector in one go.
    3. Distribute cleaned gradients back to the parameters.
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            # Create a placeholder zero gradient if it doesn't exist
            # to keep consistent parameter count
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)

    # Flatten all gradients into a single vector
    flat_grad = parameters_to_vector(grads)

    # Clean the vector in one single operation
    torch.nan_to_num_(flat_grad, nan=0.0, posinf=0.0, neginf=0.0)

    # Distribute cleaned gradients back to parameters
    vector_to_parameters(flat_grad, model.parameters())


def get_vae_scaler(config, device):
    try:
        #path = os.path.join(config.vae.pretrained, "scaling.pt")  config.json
        vae_config_path = os.path.join(config.vae.pretrained, "config.json")  
        #scaler = torch.load(path)
        with open(vae_config_path, "r") as f:
            vae_config = json.load(f)
        scaler = {
            "std": torch.tensor(vae_config.get("scaling_factor", 1.0)),
            "mean": torch.tensor(vae_config.get("shift_factor", 0.0)),
            #"mean": torch.tensor(0),
            #"std": torch.tensor(1),           
        }
        print(f"Loaded scaler from {vae_config_path}: scaling={scaler['std']}, shift={scaler['mean']}")
    except:
        scaler = {
            "mean": torch.tensor(0),
            "std": torch.tensor(1),
        }
        print(f"""
              *** 
              WARNING: VAE scaling file not found at {path}.
              Using default mean=0 and std=1.
              ***
              """)
    scaler = {k: v.to(device) for k, v in scaler.items()}
    return scaler


### Diffusion ###


def get_scheduler_type(noise_scheduler):
    if noise_scheduler.__class__.__name__ == "EDMEulerScheduler":
        scheduler_type = Scheduler.EDM
    elif noise_scheduler.__class__.__name__ == "EulerDiscreteScheduler":
        scheduler_type = Scheduler.EULER
    else:
        scheduler_type = Scheduler.OTHER  # supports DDIM / DDPM
    return scheduler_type


def get_sigmas(timesteps, noise_scheduler, n_dim=4, dtype=torch.float32):
    # TODO: revisit other sampling algorithms
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def get_timesteps(noise_scheduler, latents):
    B = latents.shape[0]
    max_step = noise_scheduler.config.num_train_timesteps
    if get_scheduler_type(noise_scheduler) == Scheduler.OTHER:
        timesteps = torch.randint(
            0,
            int(max_step),
            (B,),
            device=latents.device,
        ).long()
    else:  # EDM or EULER
        # In EDM formulation, the model is conditioned on the pre-conditioned noise levels
        # instead of discrete timesteps, so here we sample indices to get the noise levels
        # from `scheduler.timesteps`
        indices = torch.randint(0, max_step, (B,))
        timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)
    return timesteps


def scale_noisy_latents(noisy_latents, noise_scheduler, timesteps):
    sched_type = get_scheduler_type(noise_scheduler)

    if sched_type == Scheduler.EDM:
        sigmas = get_sigmas(
            timesteps, noise_scheduler, len(noisy_latents.shape), noisy_latents.dtype
        )
        output = noise_scheduler.precondition_inputs(noisy_latents, sigmas)
    elif sched_type == Scheduler.EULER:
        sigmas = get_sigmas(
            timesteps, noise_scheduler, len(noisy_latents.shape), noisy_latents.dtype
        )
        output = noisy_latents / ((sigmas**2 + 1) ** 0.5)
    else:
        output = noisy_latents
        sigma = None

    return output, sigma


def get_noise(latents, noise_scheduler=None, noise_offset=0.0):
    # compute noise
    noise = torch.randn_like(latents)

    if noise_offset > 0.0:
        noise_offset_shape = (latents.shape[0], latents.shape[1]) + (1,) * (
            len(latents.shape) - 2
        )
        noise = noise + noise_offset * torch.randn(
            noise_offset_shape, device=latents.device
        )

    return noise


def scale_model_prediction(
    model_pred,
    noise_scheduler,
    sigmas,
    noisy_l,
):
    sched_type = get_scheduler_type(noise_scheduler)
    weighting = None

    if sched_type == Scheduler.EDM:
        model_pred = noise_scheduler.precondition_outputs(noisy_l, model_pred, sigmas)
    elif sched_type == Scheduler.EULER:
        weighting = (sigmas**-2.0).float()
        if noise_scheduler.config.prediction_type == "epsilon":
            model_pred *= (-sigmas) + noisy_l
        elif noise_scheduler.config.prediction_type == "v_prediction":
            model_pred *= -sigmas / (sigmas**2 + 1) ** 0.5
            model_pred += noisy_l / (sigmas**2 + 1)
        else:
            pass
    else:
        pass

    return model_pred, weighting


def define_target(latents, noise_scheduler, noise, timesteps):
    sched_type = get_scheduler_type(noise_scheduler)

    if sched_type in [Scheduler.EDM, Scheduler.EULER]:
        target = latents
    elif noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        assert (
            noise_scheduler.config.prediction_type == "sample"
        ), f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        target = latents

    return target


def compute_loss(model_pred, target, weighting=None, mask=None):
    model_pred = model_pred.float()
    target = target.float()

    if weighting is not None:
        B = model_pred.shape[0]
        weighting = weighting.float()

        loss = weighting * ((model_pred - target) ** 2)
        loss = loss.view(B, -1).mean(dim=1)
    elif mask is not None:
        loss = F.mse_loss(model_pred, target, reduction="none")
        loss = loss * mask
        loss = loss.mean()
    else:
        loss = F.mse_loss(model_pred, target, reduction="mean")

    return loss


def sample_latents(config, latents):
    B, C, *_ = latents.shape
    if config.sample_latents and C == 2 * config.globals.latent_channels:
        mean, std = latents.chunk(2, dim=1)
        latents = torch.randn_like(mean) * std + mean
    else:
        latents = latents[:, : config.globals.latent_channels]  # take only the mean
    return latents


def scale_latents(latents, vae_scaling=None):
        #if latents.ndim == 4:
        #    v = (1, -1, 1, 1)
        #elif latents.ndim == 5:
        #    v = (1, -1, 1, 1, 1)
        #else:
        #    raise ValueError("Latents should be 4D or 5D")
    latents -= vae_scaling["mean"]
    latents *= vae_scaling["std"]

    return latents


def unscale_latents(latents, vae_scaling=None):
        #if latents.ndim == 4:
        #    v = (1, -1, 1, 1)
        #elif latents.ndim == 5:
        #    v = (1, -1, 1, 1, 1)
        #else:
        #    raise ValueError("Latents should be 4D or 5D")
    latents /= vae_scaling["std"]
    latents += vae_scaling["mean"]

    return latents


### EMA ###


def get_ema_model(denoiser):
    ema_model = diffusers.training_utils.EMAModel(
        denoiser.parameters(),
        model_cls=denoiser.__class__,
        model_config=denoiser.config,
    )
    return ema_model


def tensor_stat(tensor):
    """
    Display a summary of common statistics for a PyTorch tensor.
    """
    summary = {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "device": str(tensor.device),
        "mean": tensor.mean().item() if tensor.numel() > 0 else None,
        "std": tensor.std().item() if tensor.numel() > 1 else None,
        "variance": tensor.var().item() if tensor.numel() > 1 else None,
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
        "sum": tensor.sum().item() if tensor.numel() > 0 else None,
        "median": tensor.median().item() if tensor.numel() > 0 else None,
        "norm": tensor.norm().item() if tensor.numel() > 0 else None,
    }

    print("Tensor Summary:")
    for key, value in summary.items():
        print(f"{key:15}: {value}")
