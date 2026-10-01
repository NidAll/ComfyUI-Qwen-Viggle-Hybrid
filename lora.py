"""Native LoRA parsing with a runtime bridge for Qwen's fused SwiGLU layout."""

import copy
import functools
import json
import math

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy.weight_adapter.lora import LoRAAdapter


def _adapter_delta(adapter: LoRAAdapter, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    # Keep the adapter on CPU; only the current layer's small A/B tensors are cast.
    local = copy.copy(adapter)
    local.weights = tuple(comfy.model_management.cast_to_device(w, x.device, x.dtype) if isinstance(w, torch.Tensor) else w for w in adapter.weights)
    return local.h(x, out)


def _linear_hook(patches):
    def hook(module, inputs, output):
        for offset, adapter in patches:
            if offset is None:
                output = output + _adapter_delta(adapter, inputs[0], output)
            else:
                _, start, length = offset
                end = start + length
                part = output[..., start:end]
                part = part + _adapter_delta(adapter, inputs[0], part)
                output = torch.cat((output[..., :start], part, output[..., end:]), dim=-1)
        return output
    return hook


def _run_with_lora(linear_patches: dict, fused_down: dict, executor, *args, **kwargs):
    model = executor.class_obj
    handles = []
    gate_outputs = {}
    try:
        for name, patches in linear_patches.items():
            handles.append(model.get_submodule(name).register_forward_hook(_linear_hook(patches)))

        for name, adapter in fused_down.items():
            mlp = model.get_submodule(name)

            def capture(module, inputs, output, name=name):
                gate_outputs[name] = output

            def add_down(module, inputs, output, name=name, adapter=adapter):
                # linear_input_act bypasses out.forward; apply its LoRA after the
                # native fused kernel, using the already-patched gate/up output.
                gate, up = gate_outputs.pop(name).chunk(2, dim=-1)
                return output + _adapter_delta(adapter, F.silu(gate) * up, output)

            handles.append(mlp.gate_up.register_forward_hook(capture))
            handles.append(mlp.register_forward_hook(add_down))
        return executor(*args, **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        gate_outputs.clear()


def patch_viggle_lora(model, viggle_lora: str, strength: float):
    if not isinstance(model.model, comfy.model_base.QwenImage21):
        raise ValueError("Qwen Viggle requires an original Qwen-Image-2.1 MODEL.")
    if not viggle_lora:
        raise ValueError("Select a Viggle v0.3 LoRA from models/loras.")
    if not math.isfinite(strength):
        raise ValueError("lora_strength must be finite.")
    path = folder_paths.get_full_path_or_raise("loras", viggle_lora)
    state_dict, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    state_dict = comfy.lora_convert.convert_lora(state_dict)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    adapters = comfy.lora.load_lora(state_dict, key_map, log_missing=False)
    if not adapters:
        raise ValueError("The selected file has no LoRA targets matching this Qwen-Image-2.1 model.")

    config = json.loads((metadata or {}).get("lora_adapter_metadata", "{}"))
    if not isinstance(config, dict):
        raise ValueError("lora_adapter_metadata must be a JSON object.")
    alpha, rank = config.get("transformer.lora_alpha", 1), config.get("transformer.r", 1)
    if not isinstance(alpha, (int, float)) or not isinstance(rank, (int, float)) or not math.isfinite(alpha) or not math.isfinite(rank) or rank <= 0:
        raise ValueError("The LoRA adapter metadata requires finite alpha and a positive rank.")
    if config.get("transformer.use_rslora", False):
        raise ValueError("Rank-stabilized LoRA scaling is unsupported; select a standard Viggle v0.3 adapter.")

    linear_patches, fused_down = {}, {}
    loaded_keys = set()
    for key, adapter in adapters.items():
        if not isinstance(adapter, LoRAAdapter) or any(value is not None for value in adapter.weights[3:]):
            raise ValueError("Use a standard linear Viggle LoRA; DoRA, reshaped adapters and weight-diff patches are unsupported.")
        loaded_keys.update(adapter.loaded_keys)
        adapter.multiplier = strength * (alpha / rank if adapter.weights[2] is None else 1.0)
        weight_key, offset = key if isinstance(key, tuple) else (key, None)
        if offset is not None and offset[0] != 0:
            raise ValueError(f"Unsupported packed LoRA target: {weight_key}.")
        name = weight_key.removeprefix("diffusion_model.").removesuffix(".weight")
        parent, _, leaf = name.rpartition(".")
        if leaf == "out" and parent.endswith(".img_mlp") and model.model.diffusion_model.get_submodule(parent).fused:
            fused_down[parent] = adapter
        else:
            linear_patches.setdefault(name, []).append((offset, adapter))

    unmatched = set(state_dict) - loaded_keys
    if unmatched:
        raise ValueError(f"LoRA tensors did not match this model: {', '.join(sorted(unmatched)[:3])}.")

    # clone() shares the transformer and keeps the native dynamic-VRAM patcher.
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "qwen_viggle_hybrid_lora", functools.partial(_run_with_lora, linear_patches, fused_down))
    return patched
