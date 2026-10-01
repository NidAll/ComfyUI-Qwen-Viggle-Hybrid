import math

import comfy.samplers
import folder_paths
from comfy_api.latest import io

from .lora import patch_viggle_lora
from .sampling import sample_hybrid
from .viggle_schedule import HYBRID_9, MODES, make_schedule


class QwenViggleHybridSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="QwenViggleHybridSampler",
            display_name="Qwen 2.1 Viggle Hybrid Sampler",
            category="sampling/viggle",
            description="Viggle v0.3: 6 Turbo steps, or 7 Turbo + 2 original-base steps with no fresh refinement noise.",
            inputs=[
                io.Model.Input("model", tooltip="Original Qwen-Image-2.1 base model; use an unmerged Viggle LoRA."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent"),
                io.Combo.Input("viggle_lora", options=folder_paths.get_filename_list("loras"), tooltip="Select the Viggle v0.3 r128 or r256 adapter in models/loras."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Combo.Input("mode", options=list(MODES), default="9_step_hybrid"),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Float.Input("lora_strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, optional=True, advanced=True, tooltip="1 uses native BasicGuider. Other values use CFGGuider and negative conditioning; experimental for Viggle."),
                io.String.Input("custom_nodes", default=", ".join(str(value) for value in HYBRID_9), optional=True, advanced=True, tooltip="Custom mode only: raw descending sigmas, before the Viggle resolution shift. Final zero is automatic."),
                io.Int.Input("switch_step", default=7, min=1, max=10000, optional=True, advanced=True, tooltip="Custom mode only: Turbo transitions before switching to base. Set to the total transition count for Turbo only."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, positive, negative, latent: dict, viggle_lora: str, seed: int,
                mode: str, sampler_name: str, lora_strength: float = 1.0, cfg: float = 1.0,
                custom_nodes: str = "", switch_step: int = 7) -> io.NodeOutput:
        if sampler_name not in comfy.samplers.SAMPLER_NAMES:
            raise ValueError(f"Unsupported ComfyUI sampler: {sampler_name!r}.")
        if not math.isfinite(cfg):
            raise ValueError("cfg must be finite.")
        turbo_sigmas, base_sigmas = make_schedule(latent, mode, custom_nodes, switch_step)
        turbo_model = patch_viggle_lora(model, viggle_lora, lora_strength)
        output = sample_hybrid(model, turbo_model, positive, negative, latent, seed, sampler_name, cfg, turbo_sigmas, base_sigmas)
        return io.NodeOutput(output)
