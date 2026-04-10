# PCLM — Predictive Coding Language Model

A novel autoregressive language model built on Hierarchical Predictive Coding.
Benchmarked against a Transformer baseline. Both train. Both converge.

## Architecture

**PCLM** (`pclm.py`)
- Top-down predictions: x_{l+1} → xhat_l via causal conv
- Bottom-up carries ONLY prediction errors: e_l = x_l - xhat_l
- Iterative inference: K steps of error-reducing dynamics before output
- Precision-weighting: pi_l scales each error (learned attention analog)
- Free energy loss: CE + lambda * sum(precision-weighted error energy)
- Phase 3 hook: persistent top-level world model state across sequences

**Transformer baseline** (`transformer_baseline.py`)
- GPT-2 style: causal MHA + MLP + pre-LayerNorm
- Matched parameter budget for fair comparison

**NanoLM** (`nanolm.py`) — pure NumPy transformer, zero dependencies
- LLaMA-3 architecture: RoPE, RMSNorm, SwiGLU, GQA
- Full analytical backprop, no autograd framework
- Autograd engine from scratch (`autograd.py`)

## Benchmark Results (300 steps, same data, same optimizer)

| Metric              | PCLM      | Transformer |
|---------------------|-----------|-------------|
| Final CE loss       | 2.8716    | 2.8879      |
| Final Perplexity    | 17.67     | 17.96       |
| Parameters          | 1,370,154 | 804,864     |
| Free energy (step 1)| 1.4886    | —           |
| Free energy (final) | 0.0732    | —           |
| Energy reduction    | 95.1%     | —           |
| Precision (step 1)  | 0.993     | —           |
| Precision (final)   | 0.484     | —           |

## Quick Start

### Requires
- Python 3.10+
- PyTorch 2.x (CPU or CUDA)

```bash
pip install torch
python train_and_benchmark.py
```

### PCLM only
```python
from pclm import PCLM, PCLMConfig

cfg = PCLMConfig(
    vocab_size=4096,
    d_model=256,
    n_levels=3,
    infer_steps=2,
    lambda_energy=0.05,
)
model = PCLM(cfg)
out = model(input_ids, targets=input_ids)
loss = out["loss"]  # CE + free energy
```

### NanoLM (no dependencies)
```bash
python run_training.py
```

## Next Frontiers

1. `infer_steps` 2 → 8: more inference iterations should compress top-down context better than transformer at equal params
2. Integration proxy regularizer on top-level latent — phi-style term as training objective
3. `persistent_state=True`: persistent world model across sequence boundaries — the thing no stateless transformer can do

## Files

| File | Description |
|------|-------------|
| `pclm.py` | PCLM core — PCLevel, GRU update, precision weighting, full forward/backward |
| `transformer_baseline.py` | GPT-2 style baseline for benchmarking |
| `train_and_benchmark.py` | Head-to-head training harness + results |
| `nanolm.py` | Pure NumPy transformer — RoPE/RMSNorm/SwiGLU/GQA + analytical backprop |
| `autograd.py` | Reverse-mode autodiff engine from scratch |
| `run_training.py` | NanoLM training script |
