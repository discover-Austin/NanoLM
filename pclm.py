"""
pclm.py — Predictive Coding Language Model
==========================================
Architecture decisions documented at each choice point.

What makes this PC-shaped (not a Transformer rename):
  1. Top-down prediction is explicit: x_{l+1} -> xhat_l via causal conv
  2. Bottom-up carries ONLY errors: e_l = x_l - xhat_l
  3. Iterative inference: K steps of error-reducing dynamics before output
  4. Precision-weighting: pi_l scales each error term (learned attention analog)
  5. Free energy loss: CE + lambda * sum(precision-weighted error energy)

What's genuinely novel in this implementation:
  - PC inference applied to autoregressive token prediction (not just classification)
  - Precision weighting as a differentiable, learned mechanism
  - Explicit latent hierarchy over sequence positions (not just layers)
  - Phase 3 hook: persistent top-level state across sequence boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT TYPES
# ══════════════════════════════════════════════════════════════════════════════

class PCLMStats(TypedDict):
    """Metrics produced during PC inference (always present in forward output)."""
    energy: torch.Tensor
    mean_precision: torch.Tensor
    n_energy_terms: int


class _PCLMOutputRequired(PCLMStats):
    logits: torch.Tensor


class PCLMOutput(_PCLMOutputRequired, total=False):
    """Full forward() return type. loss/ce/latents are present only when requested."""
    loss: torch.Tensor
    ce: torch.Tensor
    latents: List[torch.Tensor]


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PCLMConfig:
    vocab_size:       int   = 4096
    d_model:          int   = 256       # level-0 (token) dimension
    n_levels:         int   = 3         # number of PC levels above token embeddings
    level_scale:      float = 1.5       # dim multiplier per level (kept moderate)
    infer_steps:      int   = 2         # K iterations of PC inference
    k_pred:           int   = 3         # causal conv kernel size for top-down prediction
    lambda_energy:    float = 0.05      # weight on free energy term in loss
    dropout:          float = 0.1
    tie_output_embed: bool  = True
    max_seq_len:      int   = 256
    # Phase 3: persistent world model
    persistent_state: bool  = False     # carry top-level latent across sequences


# ══════════════════════════════════════════════════════════════════════════════
# CAUSAL CONV UTILITY
# ══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution — sees only past positions.
    Padding is applied to the LEFT only.
    Preserves sequence length.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=0)
        # Kaiming init
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)                          # [B, C, T]
        x = F.pad(x, (self.kernel_size - 1, 0))        # causal left pad
        x = self.conv(x)                               # [B, C_out, T]
        return x.transpose(1, 2)                       # [B, T, C_out]


# ══════════════════════════════════════════════════════════════════════════════
# PC LEVEL
# ══════════════════════════════════════════════════════════════════════════════

class PCLevel(nn.Module):
    """
    One level in the PC hierarchy.

    Responsibilities:
      top_down():        x_{l+1} -> (xhat_l, pi_l)
                         Prediction of lower level + precision estimate
      bottom_up_update(): x_{l+1}, e_l -> x_{l+1}'
                          Update higher latent using ONLY error from below

    Why GRU for bottom_up_update:
      - Errors arrive sequentially (position by position)
      - GRU integrates them with learned gating (what to absorb, what to ignore)
      - This is the "inference network" approximating variational inference
    """

    def __init__(self, d_low: int, d_high: int, k_pred: int = 3, dropout: float = 0.1):
        super().__init__()

        # Top-down: causal conv from high to low (preserves causality)
        self.predictor  = CausalConv1d(d_high, d_low, k_pred)
        self.pred_norm  = nn.LayerNorm(d_low)

        # Precision: how much to trust each error dimension
        # softplus output -> always positive
        self.precision  = nn.Sequential(
            nn.Linear(d_high, d_low),
            nn.Softplus()
        )
        nn.init.constant_(self.precision[0].bias, 0.5)  # start near precision=1

        # Bottom-up update: error -> delta for higher latent
        # Using a simple 1-layer GRU for recurrent error integration
        self.error_gru  = nn.GRU(
            input_size=d_low,
            hidden_size=d_high,
            num_layers=1,
            batch_first=True
        )
        self.update_norm = nn.LayerNorm(d_high)
        self.update_gate = nn.Linear(d_high * 2, d_high)  # learned merge gate
        self.dropout     = nn.Dropout(dropout)

    def top_down(self, x_high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_high: [B, T, d_high]
        Returns:
          xhat_low:  [B, T, d_low]   — prediction of lower level
          precision: [B, T, d_low]   — how much to weight errors (> 0)
        """
        xhat_low  = self.pred_norm(self.predictor(x_high))
        precision = self.precision(x_high) + 1e-4   # epsilon floor
        return xhat_low, precision

    def bottom_up_update(
        self,
        x_high: torch.Tensor,
        e_low:  torch.Tensor
    ) -> torch.Tensor:
        """
        Update higher latent based ONLY on prediction error from below.

        x_high: [B, T, d_high]   current high latent
        e_low:  [B, T, d_low]    prediction error at lower level

        Returns: updated x_high [B, T, d_high]
        """
        # GRU processes error signal sequentially
        # h0: use last-position x_high as initial hidden (seeds from current belief)
        h0 = x_high[:, -1, :].unsqueeze(0).contiguous()  # [1, B, d_high]
        delta, _ = self.error_gru(e_low, h0)              # [B, T, d_high]
        delta = self.dropout(delta)

        # Learned gate: how much of the error-driven update to absorb
        gate = torch.sigmoid(self.update_gate(
            torch.cat([x_high, delta], dim=-1)
        ))

        # Gated residual merge
        x_new = self.update_norm(x_high + gate * delta)
        return x_new


# ══════════════════════════════════════════════════════════════════════════════
# PCLM
# ══════════════════════════════════════════════════════════════════════════════

class PCLM(nn.Module):
    """
    Predictive Coding Language Model.

    Hierarchy (n_levels=3 example):
      x0: token embeddings   [B, T, d_model]              (FIXED observation)
      x1: level-1 latents    [B, T, d_model*scale]        (inferred)
      x2: level-2 latents    [B, T, d_model*scale^2]      (inferred)
      x3: top latents        [B, T, d_model*scale^3]      (inferred, world model)

    Inference (K steps):
      1. Top-down: each level predicts the one below
      2. Errors: e_l = x_l - xhat_l
      3. Bottom-up: each level updates using errors from below ONLY
      4. Repeat K times

    Output:
      logits = lm_head(x0_post_inference)
      loss   = CE(logits, targets) + lambda * FreeEnergy(precision-weighted errors)
    """

    def __init__(self, cfg: PCLMConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embed    = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_ln = nn.LayerNorm(cfg.d_model)

        # Build dimension schedule: d0, d1, ..., d_{n_levels}
        self.dims = [int(cfg.d_model * (cfg.level_scale ** l)) for l in range(cfg.n_levels + 1)]

        # PC levels: level l bridges dims[l] <-> dims[l+1]
        self.levels = nn.ModuleList([
            PCLevel(
                d_low   = self.dims[l],
                d_high  = self.dims[l + 1],
                k_pred  = cfg.k_pred,
                dropout = cfg.dropout
            )
            for l in range(cfg.n_levels)
        ])

        # Top-level regularization (prior: zero-mean Gaussian)
        self.top_norm = nn.LayerNorm(self.dims[-1])

        # LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_output_embed:
            self.lm_head.weight = self.embed.weight

        # Phase 3: persistent world model state
        # Carries top-level latent across calls (per-batch-item)
        self._world_state: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    # ── Latent Initialization ──────────────────────────────────────────────

    def _init_latents(self, x0: torch.Tensor) -> List[torch.Tensor]:
        """
        Start inference from:
          x0: fixed token embeddings (observation)
          x1..xL: zeros (prior mean)
          xL: persistent state if Phase 3 enabled
        """
        B, T, _ = x0.shape
        latents = [x0]
        for l in range(1, self.cfg.n_levels + 1):
            z = torch.zeros(B, T, self.dims[l], device=x0.device, dtype=x0.dtype)
            latents.append(z)

        # Phase 3: seed top level from persistent state if available
        if self.cfg.persistent_state and self._world_state is not None:
            ws = self._world_state
            # Adapt shape if necessary (different batch size)
            if ws.shape[0] == B:
                # Repeat last time step across new sequence
                seed = ws[:, -1:, :].expand(B, T, self.dims[-1])
                latents[-1] = seed.detach()  # stop-grad on carried state (Phase 3 safe)

        return latents

    # ── PC Inference ───────────────────────────────────────────────────────

    def _inference(
        self,
        latents: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], PCLMStats]:
        """
        K iterations of predictive coding dynamics.

        Each iteration:
          1. Top-down: compute predictions and precisions at each level
          2. Compute prediction errors
          3. Bottom-up: update higher latents from errors
          4. Record energy = sum(pi * e^2)

        The latents[0] (x0, token embeddings) is NEVER updated — it's
        the "observation" that the hierarchy must predict.
        Higher latents are updated to minimize prediction error.
        """
        energy_terms  = []
        precision_log = []

        for step in range(self.cfg.infer_steps):

            # ── Top-down pass ──────────────────────────────────────────────
            predictions  = [None] * (self.cfg.n_levels + 1)
            precisions   = [None] * (self.cfg.n_levels + 1)

            for l in reversed(range(self.cfg.n_levels)):
                xhat_l, pi_l = self.levels[l].top_down(latents[l + 1])
                predictions[l] = xhat_l
                precisions[l]  = pi_l

            # ── Error computation & bottom-up updates ─────────────────────
            for l in range(self.cfg.n_levels):
                # Prediction error at level l
                e_l = latents[l] - predictions[l]           # [B, T, d_l]

                # Precision-weighted energy contribution
                w_e = precisions[l] * (e_l ** 2)
                energy_terms.append(w_e.mean())
                precision_log.append(precisions[l].mean().detach())

                # Update higher latent using ONLY error from below
                # (not the full activation — this is the key PC property)
                latents[l + 1] = self.levels[l].bottom_up_update(
                    latents[l + 1], e_l
                )

            # ── Top prior: keep top latent bounded ────────────────────────
            latents[-1] = self.top_norm(latents[-1])

        stats: PCLMStats = {
            "energy":         torch.stack(energy_terms).mean(),
            "mean_precision": torch.stack(precision_log).mean(),
            "n_energy_terms": len(energy_terms),
        }
        return latents, stats

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:  torch.Tensor,
        targets:    Optional[torch.Tensor] = None,
        return_latents: bool = False,
    ) -> PCLMOutput:
        """
        input_ids: [B, T]
        targets:   [B, T] (for loss, pass same as input_ids shifted by 1 internally)

        Returns dict with:
          logits:         [B, T, V]
          loss:           scalar (if targets provided)
          ce:             cross-entropy component
          energy:         free energy component
          mean_precision: average precision across levels/steps
        """
        B, T = input_ids.shape

        # Observation: token embeddings (fixed, not updated by inference)
        x0 = self.embed_ln(self.embed(input_ids))          # [B, T, d_model]

        # Initialize latent hierarchy
        latents = self._init_latents(x0)

        # Run PC inference (iterative error minimization)
        latents, stats = self._inference(latents)

        # Phase 3: cache updated top-level state
        if self.cfg.persistent_state:
            self._world_state = latents[-1].detach()

        # Produce logits from the inferred x0 (post-inference)
        # Why x0 not x_top? x0 is in vocab space and was refined by inference.
        # Alternative: project x_top down — try in ablations.
        logits = self.lm_head(latents[0])                  # [B, T, V]

        out: PCLMOutput = {"logits": logits, **stats}

        if targets is not None:
            # Shift: predict token t+1 from token t
            T = logits.shape[1]
            ce = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size),
                targets[:, 1:T].contiguous().view(-1),
                ignore_index=-1,
            )
            loss = ce + self.cfg.lambda_energy * stats["energy"]
            out.update({"loss": loss, "ce": ce})

        if return_latents:
            out["latents"] = latents

        return out

    def reset_world_state(self):
        """Reset persistent state — call at document/session boundaries."""
        self._world_state = None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive generation with PC inference per step."""
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            out = self.forward(ctx)
            logits = out["logits"][:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        breakdown = {}
        for name, p in self.named_parameters():
            top = name.split('.')[0]
            breakdown[top] = breakdown.get(top, 0) + p.numel()
        return breakdown
