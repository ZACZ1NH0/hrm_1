from typing import Optional
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Plain Transformer block (PyTorch native MHA).
    - Uses key_padding_mask (True = mask/pad) for clarity.
    """
    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_mult * hidden_size),
            nn.GELU(),
            nn.Linear(ff_mult * hidden_size, hidden_size),
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attn (use key_padding_mask for (B, S) masks)
        qkv = self.ln1(x)
        a, _ = self.attn(qkv, qkv, qkv, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        # FFN
        x = x + self.ff(self.ln2(x))
        return x


class ReasoningModule(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_mult) for _ in range(depth)
        ])

    def forward(self, hidden_states: torch.Tensor, injected: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Simple additive injection (can be replaced by gated/residual fusion later)
        x = hidden_states + injected
        for blk in self.layers:
            x = blk(x, key_padding_mask=key_padding_mask)
        return x