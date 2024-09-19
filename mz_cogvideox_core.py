

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
    "timestep_activation_fn": "silu"
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
                    torch.bfloat16) if cls.bias is not None else None
                out_dtype = torch.bfloat16

                if bias is not None:
                    o = torch._scaled_mm(
                        inn, w, out_dtype=out_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
                else:
                    o = torch._scaled_mm(
                        inn, w, out_dtype=out_dtype, scale_a=scale_input, scale_b=scale_weight)

                if isinstance(o, tuple):
                    o = o[0]

                return o.reshape((-1, x.shape[1], cls.weight.shape[0]))
        else:
            cls.to(torch.bfloat16)
            out = cls.original_forward(x.to(
                torch.bfloat16
            ))
            cls.to(original_dtype)
            return out
    else:
        return cls.original_forward(x)


import torch.nn as nn
from types import MethodType


def convert_fp8_linear(module, dtype):
    for name, module in module.named_modules():
        if isinstance(module, nn.Linear):
            module.to(dtype)
            original_forward = module.forward
            setattr(module, "original_forward", original_forward)
            setattr(module, "forward", MethodType(fp8_linear_forward, module))


def MZ_CogVideoXLoader_call(args={}):
    unet_name = args.get("unet_name")

    unet_path = folder_paths.get_full_path("unet", unet_name)

    dyn_offload_cpu_layer = args.get("dyn_offload_cpu_layer", 0)
    enable_sequential_cpu_offload = args.get(
        "enable_sequential_cpu_offload", False)

    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    comfy.model_management.soft_empty_cache()

    unet_sd = safetensors.torch.load_file(unet_path)
    unet_sd_keys = list(unet_sd.keys())
    transformer_config = cogVideoXTransformerConfig
    vae_config = cogVideoXVaeConfig
    scheduler_config = cogVideoXDDIMSchedulerConfig
    base_path = os.path.join(
        os.path.dirname(__file__),
        "configs",
    )
    # print(unet_sd_keys)
    transformer_type = ""
    if "patch_embed.proj.weight" in unet_sd_keys:
        if unet_sd["patch_embed.proj.weight"].shape == (3072, 33, 2, 2):
            transformer_config = cogVideoXTransformerConfig5B
            transformer_config["in_channels"] = 33
            vae_config = cogVideoXVaeConfig5B
            scheduler_config = cogVideoXDDIMSchedulerConfig5B
            base_path = os.path.join(
                os.path.dirname(__file__),
                "configs-Fun",
            )
            transformer_type = "fun_5b"
        else:
            raise Exception("This model is not supported")

    elif len([k for k in unet_sd_keys if "transformer_blocks.39" in k]) > 0:
        transformer_config = cogVideoXTransformerConfig5B
        vae_config = cogVideoXVaeConfig5B
        scheduler_config = cogVideoXDDIMSchedulerConfig5B
        base_path = os.path.join(
            os.path.dirname(__file__),
            "configs5b",
        )
        
    dtype = None
    weight_dtype = args.get("weight_dtype")

    transformer = None
    if weight_dtype not in ["GGUF"]:
        if transformer_type == "fun_5b":
            transformer = CogVideoXTransformer3DModelFun.from_config(
                transformer_config)
        else:
            transformer = CogVideoXTransformer3DModel.from_config(
                transformer_config)

        transformer.load_state_dict(unet_sd)

    if weight_dtype == "fp8_e4m3fn":
        dtype = torch.float8_e4m3fn
        transformer.to(dtype)
    elif weight_dtype == "fp8_e5m2":
        dtype = torch.float8_e5m2
        transformer.to(dtype)
    elif weight_dtype == "GGUF":
        dtype = torch.float8_e4m3fn
        from . import mz_gguf_loader
        import importlib
        importlib.reload(mz_gguf_loader)
        with mz_gguf_loader.quantize_lazy_load():
            if transformer_type == "fun_5b":
                transformer = CogVideoXTransformer3DModelFun.from_config(
                    transformer_config)
            else:
                transformer = CogVideoXTransformer3DModel.from_config(
                    transformer_config)
            transformer.to(dtype)
            transformer = mz_gguf_loader.quantize_load_state_dict(
                transformer, unet_sd, device="cpu")
            transformer.to(device)
    else:
        dtype = transformer.parameters().__next__().dtype

    if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
        fp8_fast_mode = args.get("fp8_fast_mode", False)
        if fp8_fast_mode:
            convert_fp8_linear(transformer, dtype)

    if dyn_offload_cpu_layer > 0:
        from .mz_dyn_cpu_offload import dyn_cpu_offload_model
        transformer = dyn_cpu_offload_model(transformer)
        transformer.register_dyn_cpu_offload_model_hooks(dyn_offload_cpu_layer)
    else:
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
        "dtype": torch.bfloat16,
        "base_path": base_path,
        "onediff": False,
        "cpu_offloading": enable_sequential_cpu_offload
    }
    return (pipeline, )
