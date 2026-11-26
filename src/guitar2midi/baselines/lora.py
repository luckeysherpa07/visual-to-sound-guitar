"""Minimal LoRA (Hu et al. 2021) for nn.Linear and nn.Conv1d layers (CP16, MMAudio fine-tune).

A frozen layer W gets a trainable low-rank correction next to it:

    y = W x + (alpha / r) * B (A x)

A maps the input down to r numbers, B maps them back up. B starts at zero, so a freshly wrapped model computes
exactly what the original model computes; training only learns the change. For a Conv1d with kernel k (MMAudio's
``ChannelLastConv1d`` layers, kernel 3) A is a Conv1d in -> r with the same kernel/padding and B a 1x1 Conv1d
r -> out, the usual LoRA form for convolutions (as in ``peft``).

    n = add_lora(model, rank=16, targets=("attn.qkv", "linear1", "ffn.w1", "ffn.w2", "ffn.w3"))
    torch.save(lora_state_dict(model), "adapters.pt")
    load_lora(model, torch.load("adapters.pt"))   # after add_lora with the same settings
    merge_lora(model)                             # fold B A into W for inference (no extra cost)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        delta = (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scale
        self.base.weight += delta.to(self.base.weight.dtype)
        return self.base


class LoRAConv1d(nn.Module):
    """For nn.Conv1d and subclasses; keeps the subclass's layout (MMAudio's ChannelLastConv1d is (B, N, C))."""

    def __init__(self, base: nn.Conv1d, rank: int, alpha: float):
        super().__init__()
        assert base.groups == 1 and base.dilation == (1,) and base.stride == (1,)
        self.base = base
        self.scale = alpha / rank
        self.channel_last = type(base).__name__ == "ChannelLastConv1d"
        self.lora_A = nn.Conv1d(base.in_channels, rank, base.kernel_size, padding=base.padding, bias=False)
        self.lora_B = nn.Conv1d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2) if self.channel_last else x
        d = self.lora_B(self.lora_A(h)) * self.scale
        return self.base(x) + (d.transpose(1, 2) if self.channel_last else d)

    @torch.no_grad()
    def merged(self) -> nn.Conv1d:
        # W'[o, i, k] = W[o, i, k] + scale * sum_r B[o, r] A[r, i, k]
        delta = torch.einsum("or,rik->oik", self.lora_B.weight[:, :, 0].float(), self.lora_A.weight.float())
        self.base.weight += (delta * self.scale).to(self.base.weight.dtype)
        return self.base


def add_lora(model: nn.Module, rank: int = 16, alpha: float | None = None, targets: tuple[str, ...] = (),
             within: tuple[str, ...] = ()) -> list[str]:
    """Freeze ``model`` and wrap every Linear/Conv1d whose name ends with one of ``targets`` (and contains one of
    ``within``, if given). Returns the wrapped layer names. alpha defaults to rank (scale 1)."""
    alpha = float(rank if alpha is None else alpha)
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, (nn.Linear, nn.Conv1d)) and any(n.endswith(t) for t in targets)
             and (not within or any(w in n for w in within))]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child)
        wrap = LoRALinear if isinstance(base, nn.Linear) else LoRAConv1d
        lora = wrap(base, rank, alpha).to(base.weight.device)
        setattr(parent, child, lora)
    return names


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if ".lora_" in n]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if ".lora_" in n}


def load_lora(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = dict(model.named_parameters())
    missing = [n for n in own if ".lora_" in n and n not in state]
    assert not missing and all(n in own for n in state), f"LoRA keys do not match ({len(missing)} missing)"
    with torch.no_grad():
        for n, v in state.items():
            own[n].copy_(v)


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base layer and remove the wrapper. Returns the number of merged layers."""
    wrapped = [n for n, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv1d))]
    for name in wrapped:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).merged())
    return len(wrapped)
