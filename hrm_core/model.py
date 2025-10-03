from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from .config import HRMCoreConfig
from .inner import HRMCoreInner


class HRMForQA(nn.Module):
    """Minimal HRM-based extractive reader for multi-hop QA."""
    def __init__(self, cfg: HRMCoreConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings (can be swapped for external encoder outputs)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.ln_in = nn.LayerNorm(cfg.hidden_size)

        # Hierarchical core
        self.inner = HRMCoreInner(cfg)

        # Span head
        self.qa_head = nn.Linear(cfg.hidden_size, 2)

    def init_states(self, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        H0 = self.inner.H_init.expand(batch_size, seq_len, -1).to(device)
        L0 = self.inner.L_init.expand(batch_size, seq_len, -1).to(device)
        return H0, L0

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,  # 1=keep, 0=pad
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, S = (inputs_embeds.size(0), inputs_embeds.size(1)) if inputs_embeds is not None else input_ids.shape
        device = (inputs_embeds.device if inputs_embeds is not None else input_ids.device)
        if inputs_embeds is not None:
            # đã có pos từ encoder → KHÔNG cộng pos_emb lần nữa
            x = self.ln_in(inputs_embeds)
        else:
            pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
            x = self.ln_in(x)
        

        # Init states
        z_H, z_L = self.init_states(B, S, device)

        # Convert HF-style attention_mask -> key_padding_mask (True means masked)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Fixed-step hierarchical reasoning
        z_H, z_L = self.inner(z_H, z_L, token_embeddings=x, key_padding_mask=key_padding_mask)

        # Span head on H-level states
        logits = self.qa_head(z_H)  # [B, S, 2]
        start_logits, end_logits = logits[..., 0], logits[..., 1]

        out = {"start_logits": start_logits, "end_logits": end_logits}

        if start_positions is not None and end_positions is not None:
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            out["loss"] = (start_loss + end_loss) / 2

        return out
