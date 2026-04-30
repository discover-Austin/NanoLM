# NanoLM

**A complete LLaMA-3 architecture transformer in 840K parameters. Built entirely in NumPy. No PyTorch. No TensorFlow. Just math.**

NanoLM is a from-scratch implementation of a modern large language model — the same architecture powering LLaMA-3 — built with nothing but NumPy. Every matrix multiply, every gradient, every optimization step is written by hand. You can read every line and know exactly what a transformer *is*.

The real architecture: RoPE, RMSNorm, SwiGLU, Grouped Query Attention, BPE tokenization, AdamW with warmup and cosine decay. The only difference between this and a frontier model is scale.

## Why This Exists

Frameworks hide the math. That's useful for production — but obscures understanding.

This is a transformer where every operation is visible, traceable, and documented: the actual tensor operations, the actual gradient flow, the actual reason RoPE works. Use PyTorch to deploy transformers; use this to understand them.

## Architecture

```
┌─────────────────────────────────────────┐
│              NanoLM (840K params)        │
├─────────────────────────────────────────┤
│  Token Embedding (tied weights)         │
│  ├── BPE Tokenizer (trained from scratch)│
│  │                                       │
│  ▼ x4 Transformer Blocks                │
│  ┌─────────────────────────────────┐     │
│  │  RMSNorm (pre-normalization)    │     │
│  │  Grouped Query Attention        │     │
│  │  ├── 4 query heads              │     │
│  │  ├── 2 KV heads (GQA 2:1)      │     │
│  │  └── RoPE positional encoding   │     │
│  │  Residual connection            │     │
│  │  RMSNorm                        │     │
│  │  SwiGLU Feed-Forward            │     │
│  │  Residual connection            │     │
│  └─────────────────────────────────┘     │
│  RMSNorm (final)                         │
│  Linear → Vocab (weight-tied)            │
└─────────────────────────────────────────┘
```

Every component matches the LLaMA-3 design:

| Component | What it does | Why it matters |
|-----------|-------------|----------------|
| **RoPE** | Encodes position as rotation in embedding space | Relative position without learned embeddings; generalizes to unseen sequence lengths |
| **RMSNorm** | Normalizes by root mean square (no mean centering) | Faster than LayerNorm, empirically equivalent |
| **SwiGLU** | `SiLU(xW₁) ⊙ xW₃` gated activation | Outperforms ReLU/GELU in practice; gating provides learned feature selection |
| **GQA** | Multiple query heads share fewer KV heads | 2-4x memory reduction at inference with minimal quality loss |
| **Weight Tying** | Input embedding = output projection (transposed) | Fewer parameters, acts as implicit regularizer |

## What's Included

```
nanolm/
├── nanolm.py          # Full model — every layer, every gradient
├── autograd.py        # Reverse-mode autodiff engine (optional reference)
├── train.py           # Training loop with logging
└── checkpoints/       # Saved weights
```

**No dependencies beyond NumPy.** That's the point.

## Quick Start

```bash
git clone https://github.com/discover-Austin/NanoLM.git
cd NanoLM
python nanolm.py
```

That's it. No `pip install`, no CUDA drivers, no environment file. It trains on a built-in corpus and generates text.

```
NanoLM — Pure NumPy Transformer
Architecture: LLaMA-style (RoPE · RMSNorm · SwiGLU · GQA)

  Parameters: 840,192
  Training on 4,821 tokens...

Epoch 1/6  loss=6.241  ppl=512.0
Epoch 2/6  loss=4.103  ppl=60.5
Epoch 3/6  loss=2.847  ppl=17.2
...

> "the quick brown fox" → "the quick brown fox jumps over neural networks..."
```

## What You'll Learn By Reading This Code

1. **How attention actually works** — not the diagram, the actual einsum that computes Q·Kᵀ/√d and why scaling prevents gradient vanishing
2. **Why RoPE uses rotation matrices** — and how sin/cos frequencies encode relative distance
3. **How backpropagation flows through a transformer** — every gradient derived analytically and commented
4. **Why SwiGLU outperforms ReLU** — the gating mechanism explained at the tensor level
5. **How BPE tokenization works from byte level up** — merges, splits, special tokens
6. **How AdamW actually differs from Adam** — decoupled weight decay vs. L2 regularization (they're not the same)

## Design Decisions

**Why NumPy and not a framework?**
Frameworks fuse operations into opaque CUDA kernels. You call `nn.MultiheadAttention` and get an answer — without seeing the reshape, the mask application, the softmax stability trick, the dropout pattern, or the gradient checkpointing. Here, you see all of it.

**Why analytical gradients instead of autograd?**
An autograd engine (included in `autograd.py` for reference) builds a computation graph and walks it backward. That's elegant but adds abstraction. The main implementation computes every gradient by hand — the same thing production ML frameworks do internally with their fused kernels.

**Why 840K parameters?**
Small enough to train on a CPU in minutes. Large enough to learn real linguistic patterns and demonstrate every architectural feature at full fidelity.

## Performance

On the built-in NLP/ML corpus (deliberately small to enable CPU training):

| Metric | Value |
|--------|-------|
| Final loss | ~1.8 |
| Perplexity | ~6.0 |
| Training time | ~3 min (CPU) |
| Memory | < 100MB |

This model will not beat GPT-4. It will teach you exactly how GPT-4 works.

## Related Work

NanoLM sits in a lineage of educational transformer implementations:

- [**nanoGPT**](https://github.com/karpathy/nanoGPT) (Karpathy) — PyTorch, GPT-2 architecture, excellent walkthrough
- [**llama2.c**](https://github.com/karpathy/llama2.c) (Karpathy) — Pure C inference, LLaMA-2
- [**micrograd**](https://github.com/karpathy/micrograd) — Scalar autograd engine

NanoLM differs in three ways:
1. **LLaMA-3 architecture** (RoPE + RMSNorm + SwiGLU + GQA), not GPT-2
2. **Pure NumPy** — no framework at all, not even for training
3. **Full training loop with analytical gradients** — not just inference

## About

Built by an independent AI researcher in Indianapolis.

## License

MIT
