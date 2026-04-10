"""
train_and_benchmark.py — Train PCLM and Transformer baseline head-to-head.

Produces:
  - Loss curves for both models (same data, same optimizer)
  - Perplexity comparison
  - Energy / precision trajectory for PCLM
  - Tokens/sec throughput comparison
  - Generation samples from both
  - Full results in JSON for further analysis
"""

import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, TypedDict

from pclm import PCLM, PCLMConfig
from transformer_baseline import TransformerLM, TransformerConfig


class _TrainingHistoryRequired(TypedDict):
    name: str
    step: List[int]
    loss: List[float]
    ce: List[float]
    ppl: List[float]
    energy: List[float]
    precision: List[float]
    tok_sec: List[float]


class TrainingHistory(_TrainingHistoryRequired, total=False):
    """Accumulates per-step metrics during training; summary fields added at end."""
    total_time_s: float
    total_tokens: int
    final_loss: float
    final_ppl: float
    avg_tok_sec: float


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CORPUS — Full text for meaningful token statistics
# ══════════════════════════════════════════════════════════════════════════════

CORPUS = """
the quick brown fox jumps over the lazy dog near the river
a stitch in time saves nine lives well spent learning
to be or not to be that is the central question of existence
all that glitters is not gold worth seeking in the world
the journey of a thousand miles begins with a single determined step
knowledge is power and power requires wisdom and responsibility to use well
it was the best of times it was the worst of times for humanity
the mind once stretched by a new idea never returns to its original dimensions
ask not what your country can do for you but what you can do
intelligence is the ability to adapt to change in an uncertain environment
artificial intelligence simulates human intelligence processes using computer systems
neural networks learn by adjusting weights through backpropagation of error signals
attention is all you need for sequence to sequence language transformation
transformers revolutionized natural language processing through self attention mechanisms
language models predict the next token given all the previous context tokens
training large models requires massive compute and careful optimization strategies
gradient descent finds the minimum of a loss function through iterative updates
the transformer architecture uses multi head attention and feed forward networks
machine learning models learn patterns from data through optimization algorithms
deep learning uses multiple layers of neural networks to learn hierarchical features
convolutional neural networks excel at image recognition and computer vision tasks
recurrent neural networks process sequential data through hidden state transitions
the vanishing gradient problem motivated development of long short term memory cells
batch normalization stabilizes training by normalizing activations across mini batches
dropout regularization prevents overfitting by randomly zeroing activations during training
residual connections allow gradients to flow through very deep networks without vanishing
the softmax function converts raw scores into a probability distribution over classes
cross entropy loss measures the distance between predicted and true probability distributions
adam optimizer combines momentum and adaptive learning rates for stable convergence
weight decay penalizes large parameter values to prevent overfitting to training data
learning rate scheduling controls how quickly the model learns throughout training
embeddings map discrete tokens to continuous vector representations in high dimensional space
cosine similarity measures the angle between two vectors in the embedding space
word vectors capture semantic relationships through distributional patterns in text
the encoder transforms input sequences into contextual representations for downstream tasks
the decoder generates output sequences token by token using the encoder representations
beam search finds approximate maximum likelihood sequences during model inference
temperature scaling controls the randomness of token sampling during text generation
top k sampling restricts generation to the k most likely tokens at each step forward
perplexity measures how well a language model predicts a held out test corpus well
rotary position embeddings encode relative position information directly in attention
grouped query attention reduces memory usage by sharing key and value heads across queries
swiglu activation combines swish and gated linear units for improved feed forward performance
root mean square normalization provides stable training without mean centering computation
tokenization converts raw text into discrete units that models can process numerically
byte pair encoding iteratively merges the most frequent character pairs into new tokens
zero shot learning generalizes to new tasks without any task specific training examples
few shot learning adapts to new tasks with only a handful of labeled training examples
instruction tuning trains models to follow natural language instructions from human users
reinforcement learning from human feedback aligns model outputs with human preferences well
the scaling hypothesis predicts that larger models trained on more data continue to improve
emergent abilities arise in large models that are completely absent in smaller model versions
chain of thought reasoning improves model performance on complex multi step reasoning problems
mechanistic interpretability studies the internal computations of neural network models
sparse attention reduces computational complexity by attending to a subset of all positions
mixture of experts scales model capacity by routing each token to specialized sub networks
predictive coding proposes that the brain generates predictions and only transmits errors
free energy minimization drives biological perception according to active inference theory
hierarchical generative models explain how the cortex processes sensory information
precision weighting controls the influence of prediction errors on belief updating processes
top down predictions flow from higher to lower levels in the cortical hierarchy structure
bottom up signals carry only prediction errors not raw sensory information to higher levels
iterative inference converges to a minimum of the variational free energy over time
world models maintain an internal representation of environmental dynamics for planning
consciousness may arise from recursive self modeling and integrated information dynamics
the hard problem of consciousness concerns why physical processes give rise to experience
integrated information theory measures consciousness by computing phi over neural systems
global workspace theory proposes a central broadcast mechanism for conscious access
attention and awareness are related but dissociable cognitive and neural processes
working memory maintains information in an active state for ongoing cognitive processing
long term potentiation strengthens synaptic connections between neurons that fire together
hebbian learning captures the principle that neurons that fire together wire together well
backpropagation through time enables learning in recurrent neural networks over sequences
the credit assignment problem asks how to attribute outcomes to past causes over time
reinforcement learning solves sequential decision making problems through reward signals
model based reinforcement learning uses an internal world model to plan ahead efficiently
curiosity driven exploration motivates agents to seek novel and surprising experiences
intrinsic motivation rewards agents for reducing prediction error and increasing knowledge
active inference unifies perception action and learning under a single free energy principle
the free energy principle states that all adaptive systems minimize their variational free energy
generative models learn to produce samples that match the distribution of training data
variational autoencoders learn disentangled latent representations through evidence lower bounds
generative adversarial networks train a generator and discriminator in a minimax game
diffusion models learn to reverse a gradual noising process applied to training examples
flow based models learn invertible transformations between data space and latent space
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# SIMPLE CHARACTER TOKENIZER (no external deps, deterministic)
# ══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """
    Character-level tokenizer.
    Simple and deterministic — focuses comparison on architecture not tokenization.
    """
    def __init__(self):
        self.ch2id: dict = {}
        self.id2ch: dict = {}

    def train(self, text: str):
        chars = sorted(set(text))
        self.ch2id = {c: i+4 for i, c in enumerate(chars)}  # reserve 0-3 for special
        self.ch2id.update({'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3})
        self.id2ch = {v: k for k, v in self.ch2id.items()}

    def encode(self, text: str) -> list:
        return [self.ch2id.get(c, 1) for c in text]

    def decode(self, ids: list) -> str:
        return ''.join(self.id2ch.get(i, '?') for i in ids
                       if i not in (0, 1, 2, 3))

    @property
    def vocab_size(self): return len(self.ch2id)


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def make_batches(token_ids: torch.Tensor, batch_size: int, seq_len: int):
    """
    Yield (input, target) pairs.
    target == input (shifted internally by model).
    """
    n      = len(token_ids)
    chunk  = seq_len + 1
    starts = list(range(0, n - chunk, seq_len // 2))  # 50% overlap for more batches

    # Shuffle
    import random
    random.shuffle(starts)

    for i in range(0, len(starts) - batch_size + 1, batch_size):
        batch_starts = starts[i:i + batch_size]
        batch = torch.stack([
            token_ids[s:s + chunk] for s in batch_starts
        ])  # [B, seq_len+1]
        yield batch[:, :-1], batch  # input, targets (both [B, seq_len])


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE TRAINING RUN
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model:      nn.Module,
    token_ids:  torch.Tensor,
    name:       str,
    steps:      int  = 300,
    batch_size: int  = 8,
    seq_len:    int  = 64,
    lr:         float = 3e-4,
    device:     str  = "cpu",
    log_every:  int  = 25,
) -> TrainingHistory:

    model = model.to(device)
    model.train()
    token_ids = token_ids.to(device)

    opt       = AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)

    history: TrainingHistory = {
        "name":      name,
        "step":      [],
        "loss":      [],
        "ce":        [],
        "ppl":       [],
        "energy":    [],
        "precision": [],
        "tok_sec":   [],
    }

    step       = 0
    t_start    = time.time()
    tokens_processed = 0

    print(f"\n{'─'*60}")
    print(f"  Training: {name}  |  params: {model.num_parameters():,}")
    print(f"{'─'*60}")

    while step < steps:
        for inp, tgt in make_batches(token_ids, batch_size, seq_len):
            if step >= steps:
                break

            t0 = time.time()
            opt.zero_grad(set_to_none=True)

            out  = model(inp, targets=tgt)
            loss = out["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            t1    = time.time()
            tps   = (batch_size * seq_len) / max(t1 - t0, 1e-6)
            tokens_processed += batch_size * seq_len
            step += 1

            # Record
            loss_val  = loss.item()
            ce_val    = out.get("ce", loss).item()
            ppl       = math.exp(min(ce_val, 20))
            energy    = out.get("energy", torch.tensor(0.0)).item()
            precision = out.get("mean_precision", torch.tensor(0.0)).item()

            history["step"].append(step)
            history["loss"].append(round(loss_val, 5))
            history["ce"].append(round(ce_val, 5))
            history["ppl"].append(round(ppl, 3))
            history["energy"].append(round(energy, 6))
            history["precision"].append(round(float(precision), 5))
            history["tok_sec"].append(round(tps, 1))

            if step % log_every == 0 or step <= 3:
                elapsed = time.time() - t_start
                print(
                    f"  step {step:4d}/{steps} | "
                    f"loss={loss_val:.4f} | "
                    f"ce={ce_val:.4f} | "
                    f"ppl={ppl:7.2f} | "
                    f"energy={energy:.4f} | "
                    f"prec={float(precision):.3f} | "
                    f"{tps:6.0f} tok/s"
                )

    total_time = time.time() - t_start
    history["total_time_s"]     = round(total_time, 2)
    history["total_tokens"]     = tokens_processed
    history["final_loss"]       = history["loss"][-1]
    history["final_ppl"]        = history["ppl"][-1]
    history["avg_tok_sec"]      = round(sum(history["tok_sec"]) / len(history["tok_sec"]), 1)

    print(f"\n  ✓ Done in {total_time:.1f}s | final loss={history['final_loss']:.4f} | ppl={history['final_ppl']:.2f}")
    return history


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark():
    print("=" * 60)
    print("  PCLM vs Transformer — Head-to-Head Benchmark")
    print("  Same data · Same optimizer · Matched parameters")
    print("=" * 60)

    device = "cpu"

    # ── Tokenize ──────────────────────────────────────────────────
    print("\n[1] Tokenizing corpus...")
    tokenizer = CharTokenizer()
    tokenizer.train(CORPUS)

    token_ids = torch.tensor(tokenizer.encode(CORPUS), dtype=torch.long)
    print(f"  Chars: {len(CORPUS):,} | Tokens: {len(token_ids):,} | Vocab: {tokenizer.vocab_size}")

    # ── Build Models ──────────────────────────────────────────────
    print("\n[2] Building models with matched parameter budgets...")

    VOCAB      = tokenizer.vocab_size
    D_MODEL    = 128
    SEQ_LEN    = 64
    BATCH      = 8
    STEPS      = 300
    LR         = 3e-4

    pclm_cfg = PCLMConfig(
        vocab_size    = VOCAB,
        d_model       = D_MODEL,
        n_levels      = 3,
        level_scale   = 1.25,
        infer_steps   = 2,
        k_pred        = 3,
        lambda_energy = 0.05,
        dropout       = 0.1,
        max_seq_len   = SEQ_LEN + 10,
    )
    pclm = PCLM(pclm_cfg)

    tf_cfg = TransformerConfig(
        vocab_size  = VOCAB,
        d_model     = D_MODEL,
        n_heads     = 4,
        n_layers    = 4,
        dropout     = 0.1,
        max_seq_len = SEQ_LEN + 10,
    )
    transformer = TransformerLM(tf_cfg)

    print(f"  PCLM params:        {pclm.num_parameters():>10,}")
    print(f"  Transformer params: {transformer.num_parameters():>10,}")
    print(f"  PCLM dims:          {pclm.dims}")

    # ── Train both ────────────────────────────────────────────────
    print("\n[3] Training...")

    pclm_history = train_model(
        pclm, token_ids, "PCLM",
        steps=STEPS, batch_size=BATCH, seq_len=SEQ_LEN, lr=LR, device=device
    )

    tf_history = train_model(
        transformer, token_ids, "Transformer",
        steps=STEPS, batch_size=BATCH, seq_len=SEQ_LEN, lr=LR, device=device
    )

    # ── Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  {'Metric':<30} {'PCLM':>12} {'Transformer':>12}")
    print(f"  {'─'*54}")

    metrics = [
        ("Final CE loss",     "final_loss",  ".4f"),
        ("Final Perplexity",  "final_ppl",   ".2f"),
        ("Avg tok/sec",       "avg_tok_sec", ".0f"),
        ("Total time (s)",    "total_time_s",".1f"),
        ("Parameters",        None,          None),
    ]

    for label, key, fmt in metrics:
        if key is None:
            pv = f"{pclm.num_parameters():,}"
            tv = f"{transformer.num_parameters():,}"
        else:
            pv = f"{pclm_history[key]:{fmt}}"
            tv = f"{tf_history[key]:{fmt}}"
        print(f"  {label:<30} {pv:>12} {tv:>12}")

    # PCLM-specific
    final_energy    = pclm_history["energy"][-1]
    final_precision = pclm_history["precision"][-1]
    init_energy     = pclm_history["energy"][0]
    print(f"\n  PCLM-specific metrics:")
    print(f"    Free energy (initial):  {init_energy:.5f}")
    print(f"    Free energy (final):    {final_energy:.5f}")
    print(f"    Energy reduction:       {(1 - final_energy/max(init_energy,1e-8))*100:.1f}%")
    print(f"    Final mean precision:   {final_precision:.4f}")

    # ── Generation Samples ────────────────────────────────────────
    print("\n[4] Generation samples...\n")

    prompts = [
        "the neural network learns",
        "attention is all you need",
        "intelligence is the",
        "the free energy",
    ]

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        if ids.shape[1] == 0:
            continue

        pclm.eval()
        transformer.eval()

        with torch.no_grad():
            p_out = pclm.generate(ids, max_new_tokens=60, temperature=0.8, top_k=10)
            t_out = transformer.generate(ids, max_new_tokens=60, temperature=0.8, top_k=10)

        p_text = tokenizer.decode(p_out[0].tolist())
        t_text = tokenizer.decode(t_out[0].tolist())

        print(f"  Prompt: '{prompt}'")
        print(f"  PCLM:        '{p_text[:120]}'")
        print(f"  Transformer: '{t_text[:120]}'")
        print()

    # ── Save Results ──────────────────────────────────────────────
    results = {
        "pclm":        pclm_history,
        "transformer": tf_history,
        "config": {
            "vocab":    VOCAB,
            "d_model":  D_MODEL,
            "seq_len":  SEQ_LEN,
            "batch":    BATCH,
            "steps":    STEPS,
            "lr":       LR,
        }
    }

    os.makedirs("/home/claude/pclm/results", exist_ok=True)
    with open("/home/claude/pclm/results/benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print("  Results saved to /home/claude/pclm/results/benchmark.json")

    # ── PCLM Precision Trajectory ─────────────────────────────────
    print("\n  PCLM Precision trajectory (first→last 5 checkpoints):")
    precs = pclm_history["precision"]
    n = len(precs)
    sample_steps = [0, n//4, n//2, 3*n//4, n-1]
    for i in sample_steps:
        print(f"    step {pclm_history['step'][i]:4d}: precision={precs[i]:.4f} | energy={pclm_history['energy'][i]:.5f}")

    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    results = run_benchmark()
