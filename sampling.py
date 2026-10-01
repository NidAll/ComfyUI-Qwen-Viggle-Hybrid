"""Two native guider runs with one progress callback and no refinement noise."""

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_extras.nodes_custom_sampler import BasicGuider, CFGGuider, DisableNoise, RandomNoise


def _sample_stage(guider, noise, sampler, sigmas, latent: dict, callback) -> dict:
    # Same primary-output path as SamplerCustomAdvanced, without its unused x0.
    prepared = latent.copy()
    prepared["samples"] = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"], latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
    samples = guider.sample(noise.generate_noise(prepared), prepared["samples"], sampler, sigmas,
                            denoise_mask=prepared.get("noise_mask"), callback=callback,
                            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise.seed)
    prepared["samples"] = samples.to(comfy.model_management.intermediate_device())
    prepared.pop("downscale_ratio_spacial", None)
    prepared.pop("downscale_ratio_temporal", None)
    return prepared


def sample_hybrid(base_model, turbo_model, positive, negative, latent: dict, seed: int,
                  sampler_name: str, cfg: float, turbo_sigmas, base_sigmas) -> dict:
    sampler = comfy.samplers.sampler_object(sampler_name)
    total_steps = len(turbo_sigmas) + len(base_sigmas) - 2
    preview_callback = latent_preview.prepare_callback(base_model, total_steps)

    def make_guider(model):
        if cfg == 1.0:
            return BasicGuider.execute(model, positive)[0]
        return CFGGuider.execute(model, positive, negative, cfg)[0]

    def turbo_callback(step, x0, x, stage_steps):
        preview_callback(step, x0, x, total_steps)

    latent = _sample_stage(make_guider(turbo_model), RandomNoise.execute(seed)[0], sampler, turbo_sigmas, latent, turbo_callback)
    if len(base_sigmas) > 1:
        offset = len(turbo_sigmas) - 1

        def base_callback(step, x0, x, stage_steps):
            preview_callback(offset + step, x0, x, total_steps)

        # DisableNoise's native seed is 0, also matching the manual graph for
        # stochastic samplers. Initial noise is zero; sampler-internal RNG stays native.
        latent = _sample_stage(make_guider(base_model), DisableNoise.execute()[0], sampler, base_sigmas, latent, base_callback)
    return latent
