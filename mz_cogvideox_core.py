

from contextlib import nullcontext
import os
import torch
import comfy.supported_models
import comfy.model_base
import comfy.ldm.flux.model
import comfy.model_patcher

import comfy.model_management
import folder_paths
import safetensors.torch

from .pipeline_cogvideox import CogVideoXPipeline
from .cogvideox_fun.transformer_3d import CogVideoXTransformer3DModel as CogVideoXTransformer3DModelFun
from .cogvideox_fun.autoencoder_magvit import AutoencoderKLCogVideoX as AutoencoderKLCogVideoXFun
from .cogvideox_fun.utils import get_image_to_video_latent, ASPECT_RATIO_512, get_closest_ratio, to_pil
from .cogvideox_fun.pipeline_cogvideox_inpaint import CogVideoX_Fun_Pipeline_Inpaint
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.schedulers import CogVideoXDDIMScheduler


cogVideoXVaeConfig = {
    "act_fn": "silu",
    "block_out_channels": [
        128,
        256,
        256,
        512
    ],
    "down_block_types": [
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D"
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 3,
    "mid_block_add_attention": True,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 256,
    "scaling_factor": 1.15258426,
    "shift_factor": None,
    "temporal_compression_ratio": 4,
    "up_block_types": [
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D"
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False
}

cogVideoXVaeConfig5B = {
    "act_fn": "silu",
    "block_out_channels": [
        128,
        256,
        256,
        512
    ],
    "down_block_types": [
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D",
        "CogVideoXDownBlock3D"
    ],
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "latents_mean": None,
    "latents_std": None,
    "layers_per_block": 3,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_height": 480,
    "sample_width": 720,
    "scaling_factor": 0.7,
    "shift_factor": None,
    "temporal_compression_ratio": 4,
    "up_block_types": [
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D",
        "CogVideoXUpBlock3D"
    ],
    "use_post_quant_conv": False,
    "use_quant_conv": False
}

cogVideoXTransformerConfig = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 16,
    "max_text_seq_length": 226,
    "norm_elementwise_affine": True,
    "norm_eps": 1e-05,
    "num_attention_heads": 30,
    "num_layers": 30,
    "out_channels": 16,
    "patch_size": 2,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "text_embed_dim": 4096,
    "time_embed_dim": 512,
    "timestep_activation_fn": "silu",
    "use_rotary_positional_embeddings": False
}

cogVideoXTransformerConfig5B = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "dropout": 0.0,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 16,
    "max_text_seq_length": 226,
    "norm_elementwise_affine": True,
    "norm_eps": 1e-05,
    "num_attention_heads": 48,
    "num_layers": 42,
    "out_channels": 16,
    "patch_size": 2,
    "sample_frames": 49,
    "sample_height": 60,
    "sample_width": 90,
    "spatial_interpolation_scale": 1.875,
    "temporal_compression_ratio": 4,
    "temporal_interpolation_scale": 1.0,
    "text_embed_dim": 4096,
    "time_embed_dim": 512,
    "timestep_activation_fn": "silu",
    "use_rotary_positional_embeddings": True
}

cogVideoXDDIMSchedulerConfig = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True,
    "sample_max_value": 1.0,
    "set_alpha_to_one": True,
    "snr_shift_scale": 3.0,
    "steps_offset": 0,
    "timestep_spacing": "linspace",
    "trained_betas": None,
}

cogVideoXDDIMSchedulerConfig5B = {
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": True,
    "sample_max_value": 1.0,
    "set_alpha_to_one": True,
    "snr_shift_scale": 1.0,
    "steps_offset": 0,
    "timestep_spacing": "linspace",
    "trained_betas": None,
}


def gen_fp8_linear_forward(cast_dtype):
    def fp8_linear_forward(cls, x):
        original_dtype = cls.weight.dtype
        if original_dtype == torch.float8_e4m3fn or original_dtype == torch.float8_e5m2:
            if len(x.shape) == 3:
                with torch.no_grad():
                    if original_dtype == torch.float8_e4m3fn:
                        inn = x.reshape(-1, x.shape[2]).to(torch.float8_e5m2)
                    else:
                        inn = x.reshape(-1, x.shape[2]).to(torch.float8_e4m3fn)
                    w = cls.weight.t()

                    scale_weight = torch.ones(
                        (1), device=x.device, dtype=torch.float32)
                    scale_input = scale_weight

                    bias = cls.bias.to(
                        cast_dtype) if cls.bias is not None else None

                    if bias is not None:
                        o = torch._scaled_mm(
                            inn, w, out_dtype=cast_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
                    else:
                        o = torch._scaled_mm(
                            inn, w, out_dtype=cast_dtype, scale_a=scale_input, scale_b=scale_weight)

                    if isinstance(o, tuple):
                        o = o[0]

                    return o.reshape((-1, x.shape[1], cls.weight.shape[0]))
            else:
                cls.to(cast_dtype)
                out = cls.original_forward(x.to(
                    cast_dtype
                ))
                cls.to(original_dtype)
                return out
        else:
            return cls.original_forward(x)
    return fp8_linear_forward


import torch.nn as nn
from types import MethodType


def convert_fp8_linear(module, dtype, cast_dtype):
    for name, module in module.named_modules():
        if isinstance(module, nn.Linear):
            module.to(dtype)
            original_forward = module.forward
            setattr(module, "original_forward", original_forward)
            setattr(module, "forward", MethodType(
                gen_fp8_linear_forward(cast_dtype), module))


def MZ_CogVideoXLoader_call(args={}):
    unet_name = args.get("unet_name")

    unet_path = folder_paths.get_full_path("unet", unet_name)

    enable_sequential_cpu_offload = args.get(
        "enable_sequential_cpu_offload", False)

    device = comfy.model_management.get_torch_device()

    comfy.model_management.soft_empty_cache()

    unet_sd = safetensors.torch.load_file(unet_path)
    unet_sd_keys = list(unet_sd.keys())

    transformer_type = ""
    if unet_sd["patch_embed.proj.weight"].shape == (3072, 33, 2, 2):
        transformer_type = "fun_5b"
    elif unet_sd["patch_embed.proj.weight"].shape == (3072, 16, 2, 2):
        transformer_type = "5b"
    elif unet_sd["patch_embed.proj.weight"].shape == (1920, 33, 2, 2):
        transformer_type = "fun_2b"
    elif unet_sd["patch_embed.proj.weight"].shape == (1920, 16, 2, 2):
        transformer_type = "2b"
    else:
        raise Exception("This model is not supported")

    is_GGUF = False
    if len([k for k in unet_sd_keys if "Q4_0_qweight" in k]) > 0:
        is_GGUF = True

    print(f"transformer type: {transformer_type}")
    print(f"GGUF: {is_GGUF}")

    transformer_config = None
    vae_config = None
    scheduler_config = None
    base_path = None

    if transformer_type.endswith("5b"):
        transformer_config = cogVideoXTransformerConfig5B
        vae_config = cogVideoXVaeConfig5B
        scheduler_config = cogVideoXDDIMSchedulerConfig5B
        base_path = os.path.join(
            os.path.dirname(__file__),
            "configs5b",
        )

        if transformer_type == "fun_5b":
            transformer_config["in_channels"] = 33
            base_path = os.path.join(
                os.path.dirname(__file__),
                "configs5b-Fun",
            )

    if transformer_type.endswith("2b"):
        transformer_config = cogVideoXTransformerConfig
        vae_config = cogVideoXVaeConfig
        scheduler_config = cogVideoXDDIMSchedulerConfig
        base_path = os.path.join(
            os.path.dirname(__file__),
            "configs",
        )
        if transformer_type == "fun_2b":
            transformer_config["in_channels"] = 33
            base_path = os.path.join(
                os.path.dirname(__file__),
                "configs2b-Fun",
            )

    weight_dtype = None
    manual_cast_dtype = None
    _weight_dtype = args.get("weight_dtype")

    if _weight_dtype == "fp8_e4m3fn":
        weight_dtype = torch.float8_e4m3fn
        manual_cast_dtype = torch.float16
    elif _weight_dtype == "fp8_e5m2":
        weight_dtype = torch.float8_e5m2
        manual_cast_dtype = torch.bfloat16
    elif _weight_dtype == "fp16":
        weight_dtype = torch.float16
        manual_cast_dtype = torch.float16
    elif _weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
        manual_cast_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
        manual_cast_dtype = torch.float32

    print(
        f"model weight dtype: {weight_dtype} manual cast dtype: {manual_cast_dtype}")

    transformer = None
    CogVideoXTransformer3DModelImp = None
    if transformer_type.startswith("fun"):
        CogVideoXTransformer3DModelImp = CogVideoXTransformer3DModelFun
    else:
        CogVideoXTransformer3DModelImp = CogVideoXTransformer3DModel

    from . import mz_gguf_loader
    import importlib
    importlib.reload(mz_gguf_loader)
    with mz_gguf_loader.quantize_lazy_load() if is_GGUF else nullcontext():
        transformer = CogVideoXTransformer3DModelImp.from_config(
            transformer_config)
        transformer.to(weight_dtype)
        if is_GGUF:
            transformer = mz_gguf_loader.quantize_load_state_dict(
                transformer, unet_sd, device="cpu", cast_dtype=manual_cast_dtype)
            transformer.to(device)
        else:
            transformer.load_state_dict(unet_sd)

    if weight_dtype == torch.float8_e4m3fn or weight_dtype == torch.float8_e5m2:
        fp8_fast_mode = args.get("fp8_fast_mode", False)
        if fp8_fast_mode:
            print("convert to fp8 linear")
            convert_fp8_linear(transformer, weight_dtype, manual_cast_dtype)

        if transformer_type.endswith("2b"):
            transformer.pos_embedding = transformer.pos_embedding.to(
                manual_cast_dtype)

    transformer.to(device)

    vae_name = args.get("vae_name")
    vae_path = folder_paths.get_full_path("vae", vae_name)
    if transformer_type == "fun_5b":
        vae = AutoencoderKLCogVideoXFun.from_config(vae_config)
    else:
        vae = AutoencoderKLCogVideoX.from_config(vae_config)
    vae_sd = safetensors.torch.load_file(vae_path)
    vae.load_state_dict(vae_sd)
    vae.to(device)
    # from .mz_dyn_cpu_offload import dyn_cpu_offload_model_vae
    # vae = dyn_cpu_offload_model_vae(vae)

    scheduler = CogVideoXDDIMScheduler.from_config(
        scheduler_config)

    if transformer_type == "fun_5b":
        pipe = CogVideoX_Fun_Pipeline_Inpaint(vae, transformer, scheduler)
    else:
        pipe = CogVideoXPipeline(vae, transformer, scheduler)

    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()

    pipeline = {
        "pipe": pipe,
        "dtype": manual_cast_dtype,
        "base_path": base_path,
        "onediff": False,
        "cpu_offloading": enable_sequential_cpu_offload,
        "scheduler_config": scheduler_config,
    }
    return (pipeline, )
