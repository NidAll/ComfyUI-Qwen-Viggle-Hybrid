# Qwen 2.1 Viggle Hybrid Sampler

A small ComfyUI extension for Viggle Turbo v0.3. One node produces a standard
`LATENT`, using native ComfyUI sampling and the original Qwen-Image-2.1 model.
No additional Python dependencies are needed.

## Installation

Copy this directory to `ComfyUI/custom_nodes/ComfyUI-Qwen-Viggle-Hybrid/` and
restart ComfyUI. Find **Qwen 2.1 Viggle Hybrid Sampler** under `sampling/viggle`.
Put the unmerged Viggle v0.3 r128 or r256 adapter in `ComfyUI/models/loras/`.

Use a current ComfyUI installation with native Qwen-Image-2.1 and the V3 node API.
The existing Viggle custom nodes are only needed for the manual comparison example.
This extension makes no network requests and downloads no weights.

## Modes

| Mode | Turbo transitions | Original-base transitions |
| --- | ---: | ---: |
| `6_step_turbo` | 6 | 0 |
| `9_step_hybrid` (default) | 7 | 2 |
| `custom` | `switch_step` | Remaining transitions |

Defaults: Euler, LoRA strength 1, CFG 1. At CFG 1 the node uses `BasicGuider`;
negative conditioning is accepted for workflow compatibility. Other CFG values
use `CFGGuider` and both conditionings. Other samplers and strengths are experiments.

The official raw nodes are:

```text
6: 1, 0.9375, 0.875, 0.75, 0.5, 0.25
9: 1, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, 1/6, 1/12
```

Both receive Viggle's resolution-dependent shift, followed by a final zero:

```text
ratio = latent.get("downscale_ratio_spacial", 16) / 16
tokens = round(latent_height * ratio) * round(latent_width * ratio)
mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
sigma(t) = exp(mu) / (exp(mu) + (1/t - 1))
```

The helper calls native `comfy.model_sampling.flux_time_shift(mu, 1, t)` in
float64 and converts the finished schedule to float32, matching Viggle. There is
no terminal stretching or second shift through a generic scheduler.

For hybrid sampling, native `SplitSigmas.execute(sigmas, 7)` returns
`sigmas[:8]` and `sigmas[7:]`. The boundary is shared: the first stage stops at
sigma 7 and the second starts there. The first sampler's **output** latent feeds
the next stage; its `denoised_output` would be the wrong continuation.

```text
Base Qwen Model
      |
      +----------------------------+
      |                            |
      v                            v
Viggle LoRA Patch              Base Model
      |                            |
      v                            |
Turbo 7 Steps                     |
      |                            |
      +-------- latent ----------->+
                                   |
                                   v
                             Base 2 Steps
                                   |
                                   v
                                LATENT
```

The last two steps use the original base model as prescribed by Viggle, to
recover fine details and improve small text. Native `RandomNoise` introduces the
initial seeded noise, respecting latent batch indices. Refinement uses native
`DisableNoise`; it adds no initial noise. Native FlowMatch inverse/scaling handles
the latent at the shared boundary. Stochastic samplers retain their own native
internal randomness, including the `DisableNoise` object's seed of 0.

## Native integration and memory

`sampling.py` follows the primary-output path of `SamplerCustomAdvanced`: native
`fix_empty_latent_channels`, `BasicGuider`/`CFGGuider`, `guider.sample`,
`sampler_object`, `RandomNoise`/`DisableNoise`, `intermediate_device`, and
`latent_preview.prepare_callback`. Masks and latent metadata are preserved. One
preview callback reports all nine steps without resetting at the model switch.
The native guider owns model loading, cleanup, conditioning and device placement.
Qwen's native sampling cleanup also clears its prefix cache between the branches.

`model.clone()` shares the transformer and preserves the model-patcher class,
including DynamicVRAM. The node never clones weights, merges the Viggle update,
or converts the base model. BF16 and native quantized models use the same path;
the text encoder and VAE are independent of this sampler.

`lora.py` is the only model-specific bridge. It uses native `load_torch_file`,
`model_lora_keys_unet`, `convert_lora`, and `load_lora` to obtain native
`LoRAAdapter` objects. A patcher `DIFFUSION_MODEL` wrapper applies their native
`h()` bypass computation. Each layer's adapter tensors use native `cast_to_device`
for that call; the complete adapter stays on CPU. This saves adapter GPU residency
at the cost of transfers. The shared model's temporary hooks are removed in
`finally`, including on interruption.

The bridge is required because ComfyUI's current debug bypass loader does not
support Qwen's packed gate/up target offsets and its fused down projection.
The bridge preserves the native fused base kernel and adds the down adapter from
the patched gate/up activation. These internal layout and adapter APIs are isolated
in one file and may need updating if upstream changes them. Standard linear LoRAs
are supported; DoRA, rank-stabilized, weight-diff and reshaped adapters fail explicitly. Use an
original base checkpoint: a premerged Turbo checkpoint cannot provide the base tail.

## Example

Connect **Load Diffusion Model → model**, native **Text Encode Qwen Image 2.1 →
positive / negative / latent**, then **sampler → VAE Decode → Save Image**. Select
the v0.3 LoRA, a seed, and `9_step_hybrid`. For edits, give the encoder a VAE and
reference images; connect its returned latent so the shift follows the target size.

`examples/hybrid_api.json` is an API prompt template. Select your own installed
filenames. `examples/manual_hybrid_api.json` expresses the equivalent graph with
Viggle's nodes, native `SplitSigmas`, two guiders, two `SamplerCustomAdvanced`
nodes, `RandomNoise`, and `DisableNoise`.

Advanced `custom_nodes` and `switch_step` apply only to `custom`; official preset
schedules ignore them. Supply strictly decreasing raw values in (0, 1], separated
by commas; fractions are accepted and zero is appended. The switch counts Turbo
transitions. Setting it to the full count disables the base stage. For example,
the six raw Turbo nodes followed by `1/6, 1/12`, with switch 6, give an experimental
6+2 run. Custom 7+3 or 8+2 runs require explicitly supplied extra nodes; they are
not official recipes. To start from less noise, use a custom schedule beginning
below 1 rather than a generic denoise control that changes the official nodes.

## Attribution

Schedules and the hybrid recipe come from [Viggle's model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).
The runtime bridge follows the behavior of [Viggle's ComfyUI source](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/comfyui/viggle_turbo.py).
Model weights retain their upstream licenses; this extension distributes no weights.
