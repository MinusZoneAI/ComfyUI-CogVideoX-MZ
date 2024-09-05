import inspect
import json
import os
import traceback
import folder_paths
import importlib
from . import mz_cogvideox_core
import comfy

AUTHOR_NAME = "MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - CogVideoX"


NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}


class MZ_CogVideoXLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"), ),
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "GGUF"],),
                "fp8_fast_mode": ("BOOLEAN", {"default": False}),
                # "dyn_offload_cpu_layer": ("INT", {"default": 0, "tooltip": "0-42,默认0不启用"}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "significantly reducing memory usage and slows down the inference"}),
            }}

    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe", )

    FUNCTION = "load"

    CATEGORY = CATEGORY_NAME

    def load(self, **kwargs):
        from . import mz_cogvideox_core
        importlib.reload(mz_cogvideox_core)
        return mz_cogvideox_core.MZ_CogVideoXLoader_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_CogVideoXLoader"] = MZ_CogVideoXLoader
NODE_DISPLAY_NAME_MAPPINGS["MZ_CogVideoXLoader"] = f"{AUTHOR_NAME} - CogVideoXLoader"
