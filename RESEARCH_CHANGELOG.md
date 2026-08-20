# 🧪 Game of the Generals (Salpakan) - Research & Experiment Log

> **System Header & Architecture Specification**
> - **Project**: Game of the Generals (Salpakan) Vectorized PyTorch RL
> - **Architecture**: `73-Token Pre-LN Transformer Encoder (d_model=192, nhead=6, layers=6, d_ff=512)`
> - **Belief Module**: Masked Log-Space Sinkhorn-Knopp Doubly-Stochastic Optimal Transport (10 iters)
> - **Self-Play**: Prioritized Fictitious Self-Play (PFSP) with 50-checkpoint pool (w_i=(1-win_rate)^1.5 + 0.05)
> - **Optimization**: GAE-PPO (gamma=0.99, lambda=0.95, clip=0.2) with Native AMP bfloat16
>
> 📌 **Protocol Reminder**: *AUTOMATIC RESEARCH PROTOCOL: This log MUST be updated after every major architectural shift (e.g. attention mechanism, memory module, positional encoding, Sinkhorn formulation) or hyperparameter sweep. Record throughput (SPS), winrate vs baselines, move quality, decisive decision making, bugs fixed, and existing limitations.*

---

## 📊 Historical Experiment Trajectory

### 🔬 `EXP-001-VECTORIZED-CUDA-PFSP` — 2026-08-20T22:48:04
- **Status**: `COMPLETED`
- **Architectural / Algorithmic Change**: Vectorized CUDA Tensor Environment, 73-token Pre-LN Transformer, Masked Sinkhorn-Knopp, and PFSP Opponent Pool
- **Throughput**: `105.0 SPS`
- **Winrate vs Heuristic**: `54.0%`
- **Losses**: $L_{pol}=0.0354$, $L_{val}=0.0099$, $L_{sink}=2.3383$, $\text{Ent}=2.91$
- **Decision Quality Assessment**: Aggressive initial probe with Privates and Spies; Flag retreats safely away from combat zones.
- **Bugs Resolved**: Fixed canonical direction inversion (dir ^ 1) and bfloat16 index_put tensor assignment.
- **Known Limitations**: Log-space Sinkhorn iterations on GPU have fixed 10-step horizon; mobile GPU power throttling during large batch backprop.

---
