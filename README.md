# 🧠 Self-Pruning Neural Network on CIFAR-10
 
> A PyTorch framework that compresses itself during training — no post-training pruning required.
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Salil-Shekhar/Self-Pruning-Neural-Network-Report/blob/main/Self-Pruning-Report.ipynb)
[![View on nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Salil-Shekhar/Self-Pruning-Neural-Network-Report/blob/main/Self-Pruning-Report.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)
 
> 📓 **Can't see plots on GitHub?** → [View the fully rendered notebook on nbviewer](https://nbviewer.org/github/Salil-Shekhar/Self-Pruning-Neural-Network-Report/blob/main/Self-Pruning-Report.ipynb)
 
---
 
## 📌 Overview
 
Deploying large neural networks in production is often constrained by memory and compute budgets. The standard solution — **pruning** — removes unimportant weights after training. This project takes a fundamentally different approach:
 
**The network learns to prune itself, dynamically, during training.**
 
This is achieved through a custom `PrunableLinear` layer that attaches a learnable **gate score** to every single weight. An L1 sparsity penalty continuously pushes non-essential gates toward zero, yielding a sparse architecture automatically — no separate pruning step needed.
 
---
 
## 🔬 Methodology
 
### 1 · Dynamic Gating
 
Every weight $w_{ij}$ is paired with a learnable gate score $g_{ij}$, passed through a Sigmoid during the forward pass:
 
$$\hat{w}_{ij} = w_{ij} \cdot \sigma(g_{ij})$$
 
A gate near `1` preserves the connection; a gate near `0` silences it entirely.
 
### 2 · Sparsity-Inducing L1 Regularization
 
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \sum_{i,j} \sigma(g_{ij})$$
 
L1 applies a **constant gradient pressure** regardless of gate magnitude. If a connection doesn't contribute enough to accuracy to overcome this pressure, its gate is driven to exactly zero — pruning that weight permanently. This is why L1 produces true sparsity while L2 does not.
 
### 3 · Two-Phase Training
 
| Phase | Epochs | What Happens |
|-------|--------|--------------|
| Gate Learning | 30 | Network trains with sparsity regulariser — unimportant gates converge to 0 |
| Fine-Tuning | 5 | Hard binary mask freezes pruned weights; remaining weights recover accuracy |
 
---
 
## 🏗️ Architecture
 
A three-layer MLP with all linear layers replaced by `PrunableLinear`:
 
```
Input (3×32×32 = 3072)
  → PrunableLinear(3072 → 512) + ReLU
  → PrunableLinear(512  → 256) + ReLU
  → PrunableLinear(256  →  10)
Output (10 classes)
```
 
---
 
## 📊 Experimental Results
 
**Config:** Adam · lr=1e-3 · CosineAnnealingLR · batch size=128
 
| Lambda (λ) | Test Accuracy | Weight Sparsity | Status | Recommendation |
|:----------:|:-------------:|:---------------:|:------:|:---------------|
| **5e-06**  | **57.42%**    | **41.59%**      | ✅ Optimal  | **Best for deployment** |
| 5e-05      | 45.85%        | 96.81%          | ⚠️ Marginal | Research only |
| 5e-04      | 10.00%        | 99.90%          | ❌ Collapse | Avoid |
 
> Uncompressed baseline: ~58–59%
 
### λ = 5e-06 · Detailed Training Log
 
| Phase | Epoch | Train Acc | CE Loss | Sparsity | Gate Mean |
|-------|------:|----------:|--------:|---------:|----------:|
| Gate Learning | 1  | 41.64% | 1.6473 | 0.00%  | 0.9770 |
| Gate Learning | 10 | 64.38% | 1.0021 | 12.19% | 0.7526 |
| Gate Learning | 20 | 80.83% | 0.5511 | 39.08% | 0.5947 |
| Gate Learning | 30 | 88.21% | 0.3691 | 41.59% | 0.5743 |
| *After Hard Mask* | — | — | — | 41.59% (Test: 55.53%) | — |
| Fine-Tuning | 1 | 85.61% | 0.4147 | — | — |
| Fine-Tuning | 5 | 88.65% | 0.3446 | — | — |
 
**✅ Final Test Accuracy: 57.42% · Final Weight Sparsity: 41.59%**
 
### λ = 5e-05 · High Compression
 
Sparsity jumps sharply at epoch 8 (0% → 78.78%) as the penalty overwhelms many gates at once. Fine-tuning only partially recovers accuracy.
 
**⚠️ Final Test Accuracy: 45.85% · Final Weight Sparsity: 96.81%**
 
### λ = 5e-04 · Collapse
 
99.90% of weights are pruned — the network loses all capacity and falls to random-chance accuracy.
 
**❌ Final Test Accuracy: 10.00% · Final Weight Sparsity: 99.90%**
 
---
 
## 🔑 Key Insight
 
**λ = 5×10⁻⁶** is the optimal compression point — it removes **41.59%** of all weights with only a **~1.6% accuracy drop** from baseline. Increasing λ further triggers model collapse as critical features are pruned before the network can compensate.
 
---
 
## 🚀 Getting Started
 
```bash
pip install torch torchvision matplotlib numpy
```
 
Click **Open in Colab** at the top to run instantly in your browser with free GPU, or clone the repo and run the notebook locally.
