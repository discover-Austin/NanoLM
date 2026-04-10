"""
nanolm.py — Self-contained NanoLM implementation
Analytically computed gradients for efficiency + correctness.
Architecture: LLaMA-style (RMSNorm, RoPE, SwiGLU, GQA)

Why analytical over symbolic autograd?
- Production ML frameworks (cuDNN) use fused analytical kernels
- Cleaner, faster, no graph overhead
- Every gradient is explicitly derived and documented
"""

import numpy as np
import json
import math
import time
import os
from enum import IntEnum
from typing import Optional, List, Tuple, Dict


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER & MODULE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class Param:
    """A trainable parameter with gradient."""
    __slots__ = ['data', 'grad', 'name']
    
    def __init__(self, data: np.ndarray, name: str = ""):
        self.data = data.astype(np.float32)
        self.grad: Optional[np.ndarray] = None
        self.name = name
    
    def zero_grad(self):
        self.grad = None
    
    def acc_grad(self, g: np.ndarray):
        """Accumulate gradient (handles first call)."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += g


class Module:
    def parameters(self) -> List[Param]:
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Param):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params
    
    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


# ══════════════════════════════════════════════════════════════════════════════
# LAYERS WITH ANALYTICAL GRADIENTS
# ══════════════════════════════════════════════════════════════════════════════

class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int):
        self.W = Param(np.random.normal(0, 0.02, (vocab_size, d_model)).astype(np.float32), "embed_W")
        self.vocab_size = vocab_size
        self.d_model = d_model
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        self._idx = idx
        return self.W.data[idx]  # (B, T, d)
    
    def backward(self, dout: np.ndarray):
        # dout: (B, T, d)
        if self.W.grad is None:
            self.W.grad = np.zeros_like(self.W.data)
        np.add.at(self.W.grad, self._idx, dout)
    
    def parameters(self):
        return [self.W]


class RMSNorm(Module):
    """RMS Normalization: y = x/rms(x) * scale"""
    
    def __init__(self, d: int, eps: float = 1e-6):
        self.scale = Param(np.ones(d, dtype=np.float32), "rmsnorm_scale")
        self.eps = eps
        self.d = d
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._rms = np.sqrt((x**2).mean(-1, keepdims=True) + self.eps)
        self._normed = x / self._rms
        return self._normed * self.scale.data
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # dout: same shape as output
        # d(scale)/d... 
        self.scale.acc_grad((dout * self._normed).reshape(-1, self.d).sum(0))
        
        # dx: backprop through x/rms(x)
        g = dout * self.scale.data
        n = self.d
        dx = (g - self._normed * (g * self._normed).mean(-1, keepdims=True)) / self._rms
        return dx
    
    def parameters(self):
        return [self.scale]


class Linear(Module):
    """y = xW^T, no bias (like LLaMA)"""
    
    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        scale = math.sqrt(2.0 / in_f)
        self.W = Param(np.random.normal(0, scale, (out_f, in_f)).astype(np.float32))
        self.b = Param(np.zeros(out_f, dtype=np.float32)) if bias else None
        self.in_f = in_f
        self.out_f = out_f
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (..., in_f)
        self._x_shape = x.shape
        self._x_flat = x.reshape(-1, self.in_f)
        out = self._x_flat @ self.W.data.T
        if self.b is not None:
            out = out + self.b.data
        return out.reshape(*self._x_shape[:-1], self.out_f)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout_flat = dout.reshape(-1, self.out_f)
        self.W.acc_grad(dout_flat.T @ self._x_flat)
        if self.b is not None:
            self.b.acc_grad(dout_flat.sum(0))
        dx = (dout_flat @ self.W.data).reshape(self._x_shape)
        return dx
    
    def parameters(self):
        p = [self.W]
        if self.b is not None:
            p.append(self.b)
        return p


# ══════════════════════════════════════════════════════════════════════════════
# ROPE POSITIONAL EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0):
    theta = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(max_seq, dtype=np.float32)
    freqs = np.outer(pos, theta)           # (T, D/2)
    return np.cos(freqs), np.sin(freqs)    # each: (T, D/2)

def rope_rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: (B, H, T, D)"""
    B, H, T, D = x.shape
    cos = cos[:T][np.newaxis, np.newaxis]  # (1,1,T,D/2)
    sin = sin[:T][np.newaxis, np.newaxis]
    x1, x2 = x[..., :D//2], x[..., D//2:]
    return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)

def rope_rotate_backward(dx_rot: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Backprop through RoPE rotation."""
    B, H, T, D = dx_rot.shape
    cos = cos[:T][np.newaxis, np.newaxis]
    sin = sin[:T][np.newaxis, np.newaxis]
    dg1, dg2 = dx_rot[..., :D//2], dx_rot[..., D//2:]
    # d(x1*cos - x2*sin)/dx1 = cos, d(x1*sin + x2*cos)/dx1 = sin
    dx1 = dg1 * cos + dg2 * sin
    dx2 = -dg1 * sin + dg2 * cos
    return np.concatenate([dx1, dx2], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# GROUPED QUERY ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class GQAttention(Module):
    """
    Grouped Query Attention with RoPE.
    Forward and full backward implemented analytically.
    """
    
    def __init__(self, d: int, nq: int, nkv: int, max_seq: int = 512):
        self.d = d
        self.nq = nq       # query heads
        self.nkv = nkv     # kv heads
        self.g = nq // nkv # groups
        self.dh = d // nq  # head dim
        self.scale = self.dh ** -0.5
        
        self.Wq = Linear(d, nq * self.dh)
        self.Wk = Linear(d, nkv * self.dh)
        self.Wv = Linear(d, nkv * self.dh)
        self.Wo = Linear(d, d)
        
        self.cos, self.sin = compute_rope_cache(self.dh, max_seq)
        
        # Build causal mask once
        self._masks = {}
    
    def _causal_mask(self, T):
        if T not in self._masks:
            self._masks[T] = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        return self._masks[T]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        H, Hkv, G, Dh = self.nq, self.nkv, self.g, self.dh
        
        q = self.Wq.forward(x).reshape(B, T, H, Dh).transpose(0,2,1,3)    # (B,H,T,Dh)
        k = self.Wk.forward(x).reshape(B, T, Hkv, Dh).transpose(0,2,1,3)  # (B,Hkv,T,Dh)
        v = self.Wv.forward(x).reshape(B, T, Hkv, Dh).transpose(0,2,1,3)  # (B,Hkv,T,Dh)
        
        q = rope_rotate(q, self.cos, self.sin)
        k = rope_rotate(k, self.cos, self.sin)
        
        # Expand KV heads to match Q heads for GQA
        k_exp = np.repeat(k, G, axis=1)  # (B,H,T,Dh)
        v_exp = np.repeat(v, G, axis=1)  # (B,H,T,Dh)
        
        # Attention scores
        S = (q @ k_exp.swapaxes(-1,-2)) * self.scale  # (B,H,T,T)
        S += self._causal_mask(T)
        
        # Softmax
        S_max = S.max(-1, keepdims=True)
        E = np.exp(S - S_max)
        A = E / (E.sum(-1, keepdims=True) + 1e-9)  # (B,H,T,T)
        
        # Context
        C = A @ v_exp  # (B,H,T,Dh)
        C_flat = C.transpose(0,2,1,3).reshape(B, T, H*Dh)  # (B,T,d)
        
        out = self.Wo.forward(C_flat)
        
        # Cache for backward
        self._cache = (x, q, k, v, k_exp, v_exp, A, C_flat, B, T)
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, q, k, v, k_exp, v_exp, A, C_flat, B, T = self._cache
        H, Hkv, G, Dh = self.nq, self.nkv, self.g, self.dh
        
        # Backward through Wo
        dC_flat = self.Wo.backward(dout)  # (B,T,H*Dh)
        dC = dC_flat.reshape(B, T, H, Dh).transpose(0,2,1,3)  # (B,H,T,Dh)
        
        # Backward through context = A @ v_exp
        dA = dC @ v_exp.swapaxes(-1,-2)           # (B,H,T,T)
        dv_exp = A.swapaxes(-1,-2) @ dC            # (B,H,T,Dh)
        
        # Backward through softmax
        # d_softmax: dS = A * (dA - sum(dA*A, dim=-1, keepdim=True))
        dS = A * (dA - (dA * A).sum(-1, keepdims=True))  # (B,H,T,T)
        dS *= self.scale
        
        # Backward through S = q @ k_exp.T
        dq = dS @ k_exp           # (B,H,T,Dh)
        dk_exp = dS.swapaxes(-1,-2) @ q  # (B,H,T,Dh)
        
        # Backward through RoPE
        dq = rope_rotate_backward(dq, self.cos, self.sin)
        dk_exp = rope_rotate_backward(dk_exp, self.cos, self.sin)
        
        # Backward through GQA repeat
        # dk_exp: (B,H,T,Dh) -> dkv: (B,Hkv,T,Dh)
        dk = dk_exp.reshape(B, Hkv, G, T, Dh).sum(2)  # sum over groups
        dv = dv_exp.reshape(B, Hkv, G, T, Dh).sum(2)
        
        # Reshape back to (B, T, Hkv*Dh) for weight backward
        dq_2d = dq.transpose(0,2,1,3).reshape(B, T, H*Dh)
        dk_2d = dk.transpose(0,2,1,3).reshape(B, T, Hkv*Dh)
        dv_2d = dv.transpose(0,2,1,3).reshape(B, T, Hkv*Dh)
        
        dx_q = self.Wq.backward(dq_2d)
        dx_k = self.Wk.backward(dk_2d)
        dx_v = self.Wv.backward(dv_2d)
        
        return dx_q + dx_k + dx_v
    
    def parameters(self):
        return (self.Wq.parameters() + self.Wk.parameters() +
                self.Wv.parameters() + self.Wo.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# SWIGLU FFN
# ══════════════════════════════════════════════════════════════════════════════

class SwiGLU(Module):
    """
    Feed-forward with SwiGLU activation.
    FFN(x) = (SiLU(W1 x) * W3 x) W2
    """
    
    def __init__(self, d: int):
        h = int(2/3 * 4 * d)
        h = ((h + 63) // 64) * 64  # round to multiple of 64
        self.W1 = Linear(d, h)  # gate
        self.W2 = Linear(h, d)  # down
        self.W3 = Linear(d, h)  # up
        self.h = h
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        g = self.W1.forward(x)        # (B,T,h) gate pre-activation
        u = self.W3.forward(x)        # (B,T,h) up projection
        
        # SiLU(g) = g * sigmoid(g)
        sig = 1.0 / (1.0 + np.exp(-np.clip(g, -88, 88)))
        silu_g = g * sig              # SiLU activation
        
        h_act = silu_g * u            # element-wise gate
        out = self.W2.forward(h_act)
        
        self._cache = (g, u, sig, silu_g, h_act)
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        g, u, sig, silu_g, h_act = self._cache
        
        dh_act = self.W2.backward(dout)  # (B,T,h)
        
        # d(silu_g * u)
        dsilu_g = dh_act * u
        du = dh_act * silu_g
        
        # d(SiLU(g)) = sigmoid(g) + g * sigmoid(g) * (1 - sigmoid(g))
        dsilu_dg = sig + g * sig * (1 - sig)
        dg = dsilu_g * dsilu_dg
        
        dx_from_W1 = self.W1.backward(dg)
        dx_from_W3 = self.W3.backward(du)
        
        return dx_from_W1 + dx_from_W3
    
    def parameters(self):
        return self.W1.parameters() + self.W2.parameters() + self.W3.parameters()


# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class TransformerBlock(Module):
    """Pre-norm block: LN -> Attn -> residual, LN -> FFN -> residual"""
    
    def __init__(self, d: int, nq: int, nkv: int, max_seq: int = 512):
        self.attn_norm = RMSNorm(d)
        self.attn = GQAttention(d, nq, nkv, max_seq)
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Attention path
        h = self.attn_norm.forward(x)
        h = self.attn.forward(h)
        x = x + h   # residual 1
        
        # FFN path
        h2 = self.ffn_norm.forward(x)
        h2 = self.ffn.forward(h2)
        x = x + h2  # residual 2
        
        self._x_pre_attn_res = None  # not caching these, use direct residual
        return x
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Backward through FFN residual: x = x_prev + ffn(norm(x_prev))
        # dout flows to both x_prev and ffn_backward
        dffn = self.ffn.backward(dout)
        dffn = self.ffn_norm.backward(dffn)
        dx = dout + dffn  # residual gradient
        
        # Backward through Attn residual
        dattn = self.attn.backward(dx)
        dattn = self.attn_norm.backward(dattn)
        dx = dx + dattn
        
        return dx
    
    def parameters(self):
        return (self.attn_norm.parameters() + self.attn.parameters() +
                self.ffn_norm.parameters() + self.ffn.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# FULL LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class NanoLM(Module):
    """
    Complete autoregressive language model with full forward + backward.
    
    Config mirrors LLaMA-3 architecture choices:
    - RMSNorm pre-normalization
    - RoPE (no positional embedding table)
    - SwiGLU FFN
    - Grouped Query Attention
    - Tied input/output embeddings
    """
    
    def __init__(self, vocab: int, d: int = 256, layers: int = 6,
                 nq: int = 8, nkv: int = 2, max_seq: int = 512):
        self.vocab = vocab
        self.d = d
        self.max_seq = max_seq
        
        self.embed = Embedding(vocab, d)
        self.blocks = [TransformerBlock(d, nq, nkv, max_seq) for _ in range(layers)]
        self.norm = RMSNorm(d)
        
        # LM head — tied weights with embedding
        self.lm_head = Linear(d, vocab)
        self.lm_head.W = self.embed.W  # weight tying
        
        n = self.num_parameters()
        print(f"NanoLM | vocab={vocab} d={d} layers={layers} nq={nq} nkv={nkv}")
        print(f"Params: {n:,} ({n/1e6:.2f}M)")
    
    def forward(self, idx: np.ndarray) -> np.ndarray:
        """idx: (B, T) -> logits: (B, T, V)"""
        x = self.embed.forward(idx)      # (B, T, d)
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        logits = self.lm_head.forward(x)  # (B, T, V)
        self._x_before_head = x
        return logits
    
    def loss_and_backward(self, idx: np.ndarray) -> float:
        """
        Full forward + backward pass.
        idx: (B, T+1) token sequences
        Returns scalar loss value.
        """
        B, Tplus1 = idx.shape
        inputs  = idx[:, :-1]  # (B, T)
        targets = idx[:, 1:]   # (B, T)
        T = inputs.shape[1]
        
        # Forward
        logits = self.forward(inputs)  # (B, T, V)
        V = self.vocab
        
        # Cross-entropy loss (numerically stable)
        logits_flat = logits.reshape(B*T, V)       # (N, V)
        targets_flat = targets.reshape(-1)          # (N,)
        
        logits_max = logits_flat.max(-1, keepdims=True)
        log_sum_exp = np.log(np.exp(logits_flat - logits_max).sum(-1, keepdims=True) + 1e-9) + logits_max
        log_probs = logits_flat - log_sum_exp       # (N, V) log softmax
        
        N = B * T
        loss = -log_probs[np.arange(N), targets_flat].mean()
        
        # Backward through cross-entropy
        probs = np.exp(log_probs)                   # (N, V) softmax probs
        dlogits = probs.copy()
        dlogits[np.arange(N), targets_flat] -= 1.0
        dlogits /= N                                # (N, V)
        dlogits = dlogits.reshape(B, T, V)          # (B, T, V)
        
        # Backward through lm_head
        dx = self.lm_head.backward(dlogits)         # (B, T, d)
        
        # Backward through final norm
        dx = self.norm.backward(dx)
        
        # Backward through transformer blocks (reversed)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        
        # Backward through embedding
        # dx: (B, T, d) - gradients for embeddings
        if self.embed.W.grad is None:
            self.embed.W.grad = np.zeros_like(self.embed.W.data)
        np.add.at(self.embed.W.grad, inputs, dx)
        
        return float(loss)
    
    def generate(self, idx: np.ndarray, n: int = 50,
                 temperature: float = 1.0, top_k: int = 40) -> np.ndarray:
        for _ in range(n):
            ctx = idx[:, -self.max_seq:]
            logits = self.forward(ctx)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Top-k
            if top_k > 0:
                thresh = np.sort(next_logits, axis=-1)[:, -top_k:-top_k+1]
                next_logits = np.where(next_logits >= thresh, next_logits, -1e9)
            
            exp_l = np.exp(next_logits - next_logits.max(-1, keepdims=True))
            probs = exp_l / (exp_l.sum(-1, keepdims=True) + 1e-9)
            
            next_tok = np.array([
                np.random.choice(self.vocab, p=probs[b])
                for b in range(idx.shape[0])
            ]).reshape(-1, 1)
            
            idx = np.concatenate([idx, next_tok], axis=1)
        return idx
    
    def parameters(self):
        params = [self.embed.W]
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        # lm_head.W is same as embed.W (tied) — don't add twice
        return params


# ══════════════════════════════════════════════════════════════════════════════
# BPE TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════

class SpecialToken(IntEnum):
    """Enumeration of reserved special token IDs."""
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3


class BPETokenizer:
    SPECIAL: Dict[str, int] = {
        '<pad>': SpecialToken.PAD,
        '<unk>': SpecialToken.UNK,
        '<bos>': SpecialToken.BOS,
        '<eos>': SpecialToken.EOS,
    }
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.id2tok: Dict[int, str] = {}
        self.merges: Dict[tuple, str] = {}
        self.trained = False
    
    def train(self, texts: List[str], target_size: int = 1000, verbose: bool = True):
        self.vocab = dict(self.SPECIAL)
        
        # Collect all chars
        for t in texts:
            for c in t.lower():
                if c not in self.vocab:
                    self.vocab[c] = len(self.vocab)
        self.vocab['</w>'] = len(self.vocab)
        
        # Word frequencies
        from collections import Counter
        word_freq = Counter()
        for t in texts:
            for w in t.lower().split():
                word_freq[w] += 1
        
        # Initial word representations
        word_repr = {}
        for w, f in word_freq.items():
            word_repr[tuple(list(w) + ['</w>'])] = f
        
        self.merges = {}
        n_merges = target_size - len(self.vocab)
        
        for i in range(n_merges):
            # Count pairs
            pairs = Counter()
            for word, freq in word_repr.items():
                for j in range(len(word)-1):
                    pairs[(word[j], word[j+1])] += freq
            if not pairs:
                break
            
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            
            merged = best[0] + best[1]
            self.merges[best] = merged
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            
            # Apply merge
            new_repr = {}
            for word, freq in word_repr.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word)-1 and word[j] == best[0] and word[j+1] == best[1]:
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_repr[tuple(new_word)] = freq
            word_repr = new_repr
            
            if verbose and (i+1) % 200 == 0:
                print(f"  Merge {i+1}/{n_merges}: '{best[0]}'+'{best[1]}' -> '{merged}'")
        
        self.id2tok = {v: k for k, v in self.vocab.items()}
        self.trained = True
        if verbose:
            print(f"  Vocabulary size: {len(self.vocab)}")
    
    def _encode_word(self, word: str) -> List[int]:
        chars = tuple(list(word) + ['</w>'])
        for pair, merged in self.merges.items():
            new = []
            j = 0
            while j < len(chars):
                if j < len(chars)-1 and chars[j] == pair[0] and chars[j+1] == pair[1]:
                    new.append(merged)
                    j += 2
                else:
                    new.append(chars[j])
                    j += 1
            chars = tuple(new)
        return [self.vocab.get(t, self.SPECIAL['<unk>']) for t in chars]
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        tokens = [self.SPECIAL['<bos>']] if add_special else []
        for word in text.lower().split():
            tokens.extend(self._encode_word(word))
        if add_special:
            tokens.append(self.SPECIAL['<eos>'])
        return tokens
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        skip = set(self.SPECIAL.values()) if skip_special else set()
        toks = [self.id2tok.get(i, '<unk>') for i in ids if i not in skip]
        return ''.join(toks).replace('</w>', ' ').strip()
    
    def vocab_size(self): return len(self.vocab)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'vocab': self.vocab, 
                       'merges': [[k[0],k[1],v] for k,v in self.merges.items()]}, f)
    
    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.vocab = d['vocab']
        self.merges = {(m[0],m[1]): m[2] for m in d['merges']}
        self.id2tok = {v: k for k, v in self.vocab.items()}
        self.trained = True


# ══════════════════════════════════════════════════════════════════════════════
# ADAMW OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 wd=0.1, clip=1.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = wd
        self.clip = clip
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
    
    def cosine_lr(self, step, warmup, total):
        if step < warmup:
            return self.lr * step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return self.lr * 0.1 + self.lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    def clip_grads(self) -> float:
        norm = math.sqrt(sum(
            float(np.sum(p.grad**2)) for p in self.params if p.grad is not None
        ))
        if norm > self.clip:
            scale = self.clip / (norm + 1e-6)
            for p in self.params:
                if p.grad is not None:
                    p.grad *= scale
        return norm
    
    def step(self, lr=None):
        self.t += 1
        lr = lr or self.lr
        norm = self.clip_grads()
        
        bc1 = 1 - self.b1**self.t
        bc2 = 1 - self.b2**self.t
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.astype(np.float32)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g**2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= lr * self.wd * p.data  # weight decay
        
        return norm
    
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
