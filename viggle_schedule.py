"""Viggle v0.3 raw nodes and Qwen's resolution-dependent FlowMatch shift."""

import math
from fractions import Fraction

import torch
from comfy.model_sampling import flux_time_shift
from comfy_extras.nodes_custom_sampler import SplitSigmas


TURBO_6 = (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25)
HYBRID_9 = (1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25, 1 / 6, 1 / 12)
MODES = ("9_step_hybrid", "6_step_turbo", "custom")


def parse_nodes(text: str) -> tuple[float, ...]:
    try:
        nodes = tuple(float(Fraction(value.strip())) for value in text.split(","))
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("custom_nodes must be comma-separated numbers or fractions, such as 1, 0.5, 1/6.") from error
    if not nodes or any(not math.isfinite(value) or not 0 < value <= 1 for value in nodes):
        raise ValueError("custom_nodes must contain finite values in (0, 1]; the final zero is appended automatically.")
    if any(a <= b for a, b in zip(nodes, nodes[1:])):
        raise ValueError("custom_nodes must be strictly decreasing.")
    return nodes


def shifted_sigmas(latent: dict, nodes: tuple[float, ...]) -> torch.Tensor:
    samples = latent["samples"]
    ratio = latent.get("downscale_ratio_spacial", 16) / 16
    tokens = round(samples.shape[-2] * ratio) * round(samples.shape[-1] * ratio)
    mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)
    # Compute in float64, then round once to float32, exactly as Viggle does.
    raw = torch.tensor(nodes, dtype=torch.float64)
    sigmas = flux_time_shift(mu, 1.0, raw)
    return torch.cat((sigmas, sigmas.new_zeros(1))).float()


def make_schedule(latent: dict, mode: str, custom_nodes: str = "", switch_step: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "9_step_hybrid":
        nodes, switch = HYBRID_9, 7
    elif mode == "6_step_turbo":
        nodes, switch = TURBO_6, 6
    elif mode == "custom":
        nodes, switch = parse_nodes(custom_nodes), switch_step
        if not isinstance(switch, int) or not 1 <= switch <= len(nodes):
            raise ValueError(f"switch_step must be between 1 and {len(nodes)} (the number of custom transitions).")
    else:
        raise ValueError(f"Unsupported Viggle mode: {mode!r}.")
    sigmas = shifted_sigmas(latent, nodes)
    # Native SplitSigmas includes the switch sigma in both outputs.
    split = SplitSigmas.execute(sigmas, switch)
    return split[0], split[1]
