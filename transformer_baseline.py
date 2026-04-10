"""
transformer_baseline.py — Minimal causal Transformer LM for benchmark comparison.

Same tokenizer, same data, same optimizer, matched parameter count.
The only variable is the model architecture.

Architecture: GPT-2 style
  - Learned positional embeddings
  - Multi-head causal self-attention
  - MLP with GELU
  - Pre-layer normalization
  - No flash attention (CPU parity with PCLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TransformerConfig:
    vocab_size: int   = 4096
    d_model:    int   = 256
    n_heads:    int   = 4
    n_layers:   int   = 4
    dropout:    float = 0.1
    max_seq_len: int  = 256
    tie_output_embed: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv    = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj   = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
                  .view(1, 1, cfg.max_seq_len, cfg.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2  = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_output_embed:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.embed(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        out = {"logits": logits}
        if targets is not None:
            T = logits.shape[1]
            ce = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                targets[:, 1:T].contiguous().view(-1),
                ignore_index=-1,
            )
            out.update({"loss": ce, "ce": ce, "energy": torch.tensor(0.0)})

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx    = input_ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(ctx)["logits"][:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
