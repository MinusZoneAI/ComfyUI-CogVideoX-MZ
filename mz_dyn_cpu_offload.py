
import logging
import time
from types import MethodType

import torch

logger = logging.getLogger(__name__)


def transformer_blocks_to_cpu(m, layer_start=0, layer_size=-1):
    if layer_size == -1:
        m.transformer_blocks.to("cpu")
    else:
        m.transformer_blocks[layer_start:layer_start +
                             layer_size].to("cpu")
    # torch.cuda.empty_cache()
    # gc.collect()


def transformer_blocks_to_cuda(m, layer_start=0, layer_size=-1):
    if layer_size == -1:
        m.transformer_blocks.to("cuda")
    else:
        m.transformer_blocks[layer_start:layer_start +
                             layer_size].to("cuda")


def dyn_cpu_offload_model(model):

    def generate_transformer_blocks_forward_hook(cls, layer_start, layer_size):
        def pre_blocks_forward_hook(module, inp):
            # 当前显存占用
            start_time = time.time()
            start_vram = torch.cuda.memory_allocated() / 1024 / 1024
            start_max_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

            if layer_start > 0:
                transformer_blocks_to_cpu(model, layer_start=0,
                                          layer_size=layer_start)
            else:
                transformer_blocks_to_cpu(model)
            transformer_blocks_to_cuda(model, layer_start=layer_start,
                                       layer_size=layer_size)

            # 当前显存占用
            end_vram = torch.cuda.memory_allocated() / 1024 / 1024
            end_max_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
            logger.info(
                f"transformer_blocks_forward_hook {layer_start} -> {layer_start+layer_size} " +
                f"Current VRAM: {round(start_vram, 2)}MB -> {round(end_vram, 2)}MB {round(end_vram-start_vram, 2)}MB " +
                f"Max VRAM: {round(start_max_vram, 2)}MB -> {round(end_max_vram, 2)}MB {round(end_max_vram-start_max_vram, 2)}MB " +
                f"T:{round(time.time()-start_time, 2)}s")

            return inp

        return pre_blocks_forward_hook

    def register_hooks(cls, l):
        cls.all_dyn_cpu_offload_handles = []

        transformer_blocks_depth = len(cls.transformer_blocks)
        steps = l
        for i in range(0, transformer_blocks_depth, steps):
            s = steps
            if i + s > transformer_blocks_depth:
                s = transformer_blocks_depth - i
            pre_hook = generate_transformer_blocks_forward_hook(
                cls, i, s)
            _handle = cls.transformer_blocks[i].register_forward_pre_hook(
                pre_hook)
            cls.all_dyn_cpu_offload_handles.append(_handle)

        cls.patch_embed.to("cuda")
        cls.embedding_dropout.to("cuda")
        cls.time_proj.to("cuda")
        cls.time_embedding.to("cuda")
        cls.norm_final.to("cuda")
        cls.norm_out.to("cuda")
        cls.proj_out.to("cuda")

    setattr(model, "register_dyn_cpu_offload_model_hooks",
            MethodType(register_hooks, model))

    def unregister_hooks(self):
        for hook in self.all_dyn_cpu_offload_handles:
            hook.remove()
        del self.all_dyn_cpu_offload_handles

    setattr(model, "unregister_dyn_cpu_offload_model_hooks",
            MethodType(unregister_hooks, model))

    setattr(model, "th_to", model.to)

    def to(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, torch.device):
                continue
            if isinstance(arg, str):
                if arg == "cpu" or arg == "cuda":
                    continue

            self.th_to(arg)
        logger.info(f"model.to {args} {kwargs}")
        return self

    model.to = MethodType(to, model)
    return model
