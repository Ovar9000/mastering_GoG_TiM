# Mastering Game of the Generals (Salpakan) with Tensor Environment & Transformer PFSP

A fully vectorized, GPU-accelerated PyTorch training pipeline for **Game of the Generals (Salpakan)** using:
- **Vectorized GPU Environment (`tensor_generals_env.py`)**: Simulates thousands of parallel games concurrently on CUDA with zero CPU transfers.
- **Board Transformer Policy & Value Network (`board_transformer.py`)**: Spatial-temporal transformer architecture for board representation and action-masking policy output.
- **Prioritized Fictitious Self-Play (`train_pfsp.py`)**: Multi-agent reinforcement learning self-play framework to prevent circular strategies.
- **End-to-End Test Suite (`test_pipeline.py`)**: Unit tests and benchmarking scripts.

## Project Structure

```
├── board_transformer.py     # Transformer-based Actor-Critic architecture
├── tensor_generals_env.py   # Vectorized GPU-native Game of the Generals environment
├── train_pfsp.py            # Prioritized Fictitious Self-Play training loop
├── test_pipeline.py         # Verification and benchmark pipeline
└── .gitignore               # Ignored files & directories
```

## Getting Started

### Requirements
- Python 3.8+
- PyTorch (CUDA recommended for vectorized batch simulation)

### Running Tests
```bash
python test_pipeline.py
```

### Running Training
```bash
python train_pfsp.py
```
