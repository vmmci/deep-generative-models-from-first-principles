# Deep Generative Models from First Principles

A mathematically rigorous, theory-to-code study of deep generative models, with emphasis on probabilistic foundations, derivations, optimization objectives, and PyTorch implementations.

> **Derive it. Interpret it. Implement it. Test it.**

## About This Repository

This repository documents my study of deep generative models from a mathematical and probabilistic perspective.

The goal is not only to implement generative models, but to understand how their learning objectives arise, what assumptions they rely on, and how the mathematical formulation translates into computational algorithms.

For each major topic, the workflow is:

**Mathematical formulation → Derivation → Interpretation → Implementation → Experiment**

The material is primarily inspired by the **Deep Generative Models** course from IIT Madras, supplemented by original research papers and additional references.

---

## Mathematical Focus

The repository emphasizes topics such as:

- Probability distributions and probabilistic modeling
- Maximum likelihood estimation
- KL divergence and f-divergences
- Variational representations of divergences
- Variational inference
- Latent-variable models
- Evidence Lower Bound (ELBO)
- Adversarial objectives
- Autoregressive likelihood factorization
- Diffusion processes
- Score matching
- Stochastic generative processes
- Preference optimization and alignment objectives

Whenever possible, important objectives are derived mathematically before being implemented.

---

## Topics

### 01 — Mathematical Foundations
- [ ] Generative modeling problem formulation
- [ ] f-divergences
- [ ] Variational representation of divergences
- [ ] Variational divergence minimization
- [ ] Forward propagation
- [ ] Backpropagation
- [ ] PyTorch foundations

### 02 — Variational Inference & VAEs
- [ ] Latent-variable models
- [ ] Variational inference
- [ ] ELBO derivation
- [ ] Reparameterization trick
- [ ] VAE implementation
- [ ] Experiments

### 03 — Generative Adversarial Networks
- [ ] GAN objective
- [ ] Connection to divergence minimization
- [ ] Generator and discriminator optimization
- [ ] Training dynamics
- [ ] PyTorch implementation

### 04 — Autoregressive Models
- [ ] Probability factorization
- [ ] Maximum likelihood training
- [ ] Autoregressive generation

### 05 — Diffusion Models
- [ ] Forward diffusion process
- [ ] Reverse process
- [ ] Variational formulation
- [ ] Noise prediction objective
- [ ] DDPM implementation

### 06 — Score-Based Generative Models
- [ ] Score functions
- [ ] Score matching
- [ ] Denoising score matching
- [ ] Connection to diffusion models

### 07 — State Space Models

### 08 — LLM Alignment
- [ ] Reinforcement learning foundations
- [ ] PPO
- [ ] Preference modeling
- [ ] DPO
- [ ] Mathematical relationship between objectives

---

## Study Method

Each technical notebook is organized around the following structure:

1. **Problem Setup**
2. **Mathematical Formulation**
3. **Assumptions**
4. **Derivation**
5. **Interpretation**
6. **PyTorch Implementation**
7. **Numerical Experiment**
8. **Observations**

The emphasis is on connecting equations directly to executable implementations rather than treating theory and code as separate subjects.

---

## Repository Structure

```text
deep-generative-models-from-first-principles/
│
├── 01-foundations/
├── 02-variational-inference-and-vaes/
├── 03-gans/
├── 04-autoregressive-models/
├── 05-diffusion-models/
├── 06-score-based-models/
├── 07-state-space-models/
├── 08-llm-alignment/
│
├── experiments/
├── paper-notes/
├── references.md
└── README.md



References

Primary learning resource:

IIT Madras — Deep Generative Models

 Status

🚧 Work in progress

This repository is continuously updated as I progress through the mathematical foundations, implementations, experiments, and related research literature.

