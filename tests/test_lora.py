"""LoRA wrapper: identical output at init, merge gives the same output as the adapter, only adapters train."""

import torch
from torch import nn

from guitar2midi.baselines import lora


class ChannelLastConv1d(nn.Conv1d):  # as in MMAudio: input (B, N, C)
    def forward(self, x):
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.linear1 = ChannelLastConv1d(48, 16, 3, padding=1)
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(self.linear1(self.qkv(x)))


def test_lora_identity_merge_and_trainable():
    torch.manual_seed(0)
    m = Block()
    x = torch.randn(2, 10, 16)
    ref = m(x).detach()
    names = lora.add_lora(m, rank=4, targets=("qkv", "linear1"))
    assert names == ["qkv", "linear1"]
    assert torch.allclose(m(x), ref)  # B = 0 -> original model
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)
    with torch.no_grad():
        for p in lora.lora_parameters(m):
            p.normal_(0, 0.1)
    out = m(x).detach()
    assert not torch.allclose(out, ref, atol=1e-3)
    state = lora.lora_state_dict(m)
    assert lora.merge_lora(m) == 2
    assert isinstance(m.qkv, nn.Linear) and isinstance(m.linear1, ChannelLastConv1d)
    assert torch.allclose(m(x), out, atol=1e-5)
    torch.manual_seed(0)
    m3 = Block()
    lora.add_lora(m3, rank=4, targets=("qkv", "linear1"))
    lora.load_lora(m3, state)
    assert torch.allclose(m3(x), out, atol=1e-5)
