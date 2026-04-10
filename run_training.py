"""
run_training.py — Train NanoLM from scratch
"""

import numpy as np
import math
import time
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nanolm import NanoLM, BPETokenizer, AdamW

CORPUS = """
the quick brown fox jumps over the lazy dog
a stitch in time saves nine lives well spent
to be or not to be that is the question
all that glitters is not gold worth seeking
the journey of a thousand miles begins with one step
knowledge is power and power requires responsibility
it was the best of times it was the worst of times
the mind stretched by a new idea never returns to its original form
ask not what your country can do for you
intelligence is the ability to adapt to change
artificial intelligence simulates human intelligence with computers
neural networks learn by adjusting weights through backpropagation
attention is all you need for sequence transformation
transformers revolutionized natural language processing through self attention
language models predict the next token from context
training large models requires careful optimization strategies
gradient descent finds the minimum of a loss function
the transformer uses multi head attention and feed forward networks
machine learning finds patterns in data through optimization
deep learning uses multiple layers to learn hierarchical features
the vanishing gradient problem motivated long short term memory networks
batch normalization stabilizes training by normalizing activations
dropout prevents overfitting by randomly zeroing activations during training
residual connections allow gradients to flow through deep networks
softmax converts raw scores into a probability distribution over outputs
cross entropy loss measures distance between predicted and true distributions
adam combines momentum and adaptive learning rates for stable convergence
weight decay penalizes large parameters to prevent overfitting
learning rate scheduling controls how quickly the model learns
embeddings map discrete tokens to continuous vector representations
the encoder transforms input sequences into contextual representations
the decoder generates output sequences token by token
beam search finds approximate maximum likelihood sequences during inference
temperature scaling controls randomness during text generation sampling
perplexity measures how well a model predicts held out test data
rotary position embeddings encode relative position in attention
grouped query attention reduces memory by sharing key and value heads
swiglu activation combines swish and gated linear units for transformers
root mean square normalization provides stable training without mean centering
tokenization converts text into discrete units for model processing
byte pair encoding merges frequent character pairs into new tokens
zero shot learning generalizes to new tasks without task specific training
few shot learning adapts to new tasks with only a handful of examples
instruction tuning trains models to follow natural language directions
reinforcement learning from human feedback aligns models with preferences
the scaling hypothesis predicts larger models trained on more data improve
emergent abilities appear in large models that are absent in smaller ones
chain of thought reasoning improves performance on complex multi step problems
mechanistic interpretability studies the internal computations of neural networks
sparse attention reduces complexity by attending to a subset of positions
mixture of experts routes tokens to specialized sub networks for efficiency
"""

def make_batches(tokens, batch_size, seq_len):
    """Yield random batches of (seq_len+1) token windows."""
    n = len(tokens)
    chunk = seq_len + 1
    indices = list(range(0, n - chunk, seq_len))
    np.random.shuffle(indices)
    
    for i in range(0, len(indices) - batch_size + 1, batch_size):
        batch_idx = indices[i:i+batch_size]
        batch = np.stack([tokens[j:j+chunk] for j in batch_idx])
        yield batch

def main():
    np.random.seed(42)
    os.makedirs("/home/claude/nlp_engine/checkpoints", exist_ok=True)
    
    print("=" * 65)
    print("NanoLM — Pure NumPy Transformer")
    print("Architecture: LLaMA-style (RoPE · RMSNorm · SwiGLU · GQA)")
    print("=" * 65)
    
    # ── Tokenizer ──────────────────────────────────────────────────
    print("\n[1] Training BPE tokenizer...")
    lines = [l.strip() for l in CORPUS.strip().split('\n') if l.strip()]
    tok = BPETokenizer()
    tok.train(lines, target_size=600, verbose=True)
    
    # ── Tokenize ───────────────────────────────────────────────────
    print("\n[2] Tokenizing corpus...")
    all_tokens = []
    for line in lines:
        all_tokens.extend(tok.encode(line, add_special=True))
    tokens = np.array(all_tokens, dtype=np.int32)
    print(f"  Total tokens: {len(tokens):,}")
    
    # ── Model ──────────────────────────────────────────────────────
    print("\n[3] Initializing model...")
    V = tok.vocab_size()
    model = NanoLM(
        vocab=V,
        d=128,
        layers=4,
        nq=4,
        nkv=2,
        max_seq=64,
    )
    
    # ── Training setup ─────────────────────────────────────────────
    BATCH    = 4
    SEQ_LEN  = 48
    LR       = 1e-3
    EPOCHS   = 6
    
    params = model.parameters()
    opt = AdamW(params, lr=LR, wd=0.1, clip=1.0)
    
    batches_per_epoch = max(1, (len(tokens) - SEQ_LEN - 1) // ((SEQ_LEN+1) * BATCH))
    total_steps = EPOCHS * batches_per_epoch
    warmup = max(1, total_steps // 20)
    
    print(f"\n  Batches/epoch: ~{batches_per_epoch}")
    print(f"  Total steps:   ~{total_steps}")
    print(f"  Warmup steps:  {warmup}")
    
    # ── Training loop ──────────────────────────────────────────────
    print("\n[4] Training...\n" + "-"*65)
    
    step = 0
    best_loss = float('inf')
    history = []
    
    for epoch in range(1, EPOCHS+1):
        losses = []
        t0 = time.time()
        
        for batch in make_batches(tokens, BATCH, SEQ_LEN):
            opt.zero_grad()
            
            loss = model.loss_and_backward(batch)
            lr = opt.cosine_lr(step, warmup, total_steps)
            gnorm = opt.step(lr=lr)
            
            losses.append(loss)
            history.append({'step': step, 'loss': loss, 'lr': lr})
            step += 1
            
            if step % 5 == 0 or step <= 3:
                ppl = math.exp(min(loss, 20))
                print(f"  ep{epoch} step{step:4d}/{total_steps} | "
                      f"loss={loss:.4f} ppl={ppl:7.2f} | "
                      f"lr={lr:.2e} gnorm={gnorm:.3f}")
        
        avg = float(np.mean(losses))
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        print(f"\n  ── Epoch {epoch} avg loss={avg:.4f} ppl={ppl:.2f} ({elapsed:.1f}s)")
        
        if avg < best_loss:
            best_loss = avg
            tok.save("/home/claude/nlp_engine/checkpoints/tokenizer.json")
            print(f"  ✓ Best so far saved")
        
        # Sample generation
        print(f"\n  Sample generations:")
        for prompt in ["neural networks learn", "the transformer", "intelligence"]:
            try:
                ids = np.array([tok.encode(prompt, add_special=False)], dtype=np.int32)
                if ids.shape[1] == 0:
                    continue
                out = model.generate(ids, n=30, temperature=0.8, top_k=20)
                text = tok.decode(out[0].tolist())
                print(f"  > '{prompt}' → '{text[:100]}'")
            except Exception as e:
                print(f"  > [gen error: {e}]")
        print("-"*65)
    
    # ── Final stats ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"DONE — Best loss: {best_loss:.4f} | PPL: {math.exp(min(best_loss,20)):.2f}")
    print(f"{'='*65}")
    
    # Save history
    with open("/home/claude/nlp_engine/checkpoints/history.json", 'w') as f:
        json.dump(history, f)
    
    print("\n🚀 Optimization Roadmap to Surpass LLMs:")
    print("  Phase 1 — Architecture (you are here)")
    print("    ✓ RoPE, RMSNorm, SwiGLU, GQA — same stack as LLaMA-3")
    print("    ✓ Full backprop with analytical gradients")
    print("    ✓ AdamW + cosine warmup")
    print()
    print("  Phase 2 — Scale")
    print("    → Increase d_model: 128 → 512 → 2048 → 4096")
    print("    → More layers: 4 → 12 → 32")
    print("    → Train on real datasets (Wikipedia, books, code)")
    print()
    print("  Phase 3 — Efficiency")
    print("    → Flash Attention (fused attention kernel)")
    print("    → KV Cache (skip recomputing past tokens)")
    print("    → Gradient checkpointing (trade compute for memory)")
    print()
    print("  Phase 4 — Beyond Transformers")
    print("    → Mamba/SSM architecture (linear time, infinite context)")
    print("    → Mixture of Experts (sparse activation, 10x params)")
    print("    → Continuous thought (latent reasoning before output)")
    
    return model, tok, history

if __name__ == "__main__":
    model, tok, history = main()
