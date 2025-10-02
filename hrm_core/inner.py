from typing import Optional, Tuple
import torch
import torch.nn as nn
from .config import HRMCoreConfig
from .blocks import ReasoningModule


class HRMCoreInner(nn.Module):
    """Two-level fixed-cycle hierarchical reasoning core (H supervises L)."""
    def __init__(self, cfg: HRMCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.H_level = ReasoningModule(cfg.hidden_size, cfg.num_heads, cfg.ff_mult, cfg.H_layers)
        self.L_level = ReasoningModule(cfg.hidden_size, cfg.num_heads, cfg.ff_mult, cfg.L_layers)

        # Learnable initial latent states
        self.H_init = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.L_init = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))

    def forward(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        token_embeddings: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One outer pass comprising H/L cycles.
        Shapes:
          z_H, z_L: [B, S, D]
          token_embeddings: [B, S, D]
          key_padding_mask: [B, S] with True for PAD/ignored positions
        """
        x_H, x_L = z_H, z_L

        for _ in range(self.cfg.H_cycles):
            for _ in range(self.cfg.L_cycles):
                # L conditions on H + inputs
                x_L = self.L_level(x_L, injected=x_H + token_embeddings, key_padding_mask=key_padding_mask)
            # H updates conditioned on L summary
            x_H = self.H_level(x_H, injected=x_L, key_padding_mask=key_padding_mask)

        return x_H, x_L