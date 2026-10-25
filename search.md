I'll look for the main papers on this, including any recent ones.

Yes, this is an active research area, usually called **symbolic distillation**: train a network, then fit a closed-form equation to what it learned. The main papers I found:

- **Cranmer et al. (2020), "Discovering Symbolic Models from Deep Learning with Inductive Biases".** They train a graph neural network with sparse latent representations, then apply symbolic regression to its components. They recovered known force laws and Hamiltonians, and found a new formula for dark matter concentration. This is the foundational paper.
- **SymTorch (Tan, Soubki, Cranmer, 2026).** A library that wraps network components, records their inputs and outputs, and fits human-readable equations with PySR. It recovers n-body forces from GNNs, closed-form PDE/ODE solutions from physics-informed networks, and Lorenz dynamics. It can also replace transformer MLP layers with symbolic surrogates for some speedup. This is the most practical starting point if you want to try it yourself.
- **Kolmogorov-Arnold Networks (KAN, KAN 2.0).** These are built to be readable. Each learned activation can be parsed into a compact symbolic form. KAN 2.0 adds tools for discovering symbolic formulas and converting networks into tree graphs. A January 2026 paper used KANs to recover the Gell-Mann-Okubo baryon mass relations.
- **Symbolic-KAN (2026).** It builds discrete symbolic structure into the network and recovers correct terms and governing structures on regression and dynamical-system tasks.
- **Network dynamics (Nature Communications, 2025).** The authors fit neural networks to network-dynamics data, then use a pre-trained symbolic regression model to turn them into equations.

The common recipe is to train a network on the task, then fit symbolic regression (usually PySR) to the inputs and outputs of its components. This works best when the underlying law is simple and the network has an inductive bias that keeps its parts interpretable. It is much harder for large, general networks.

If you want, I can narrow this to a specific domain (physics, dynamical systems, LLMs) or set up a small SymTorch example.

Sources:
- [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287v2)
- [SymTorch: Symbolic Distillation of Neural Networks](https://arxiv.org/pdf/2602.21307)
- [SymTorch (abstract)](https://arxiv.org/abs/2602.21307)
- [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205v1)
- [Structured KAN Neural ODEs](https://arxiv.org/pdf/2506.18339)
- [Discovering the Gell-Mann-Okubo Formula with KANs](https://arxiv.org/abs/2601.01495v1)
- [Symbolic–KAN](https://sotaverified.org/papers/260323854)
- [Learning interpretable network dynamics via universal neural symbolic regression](https://www.nature.com/articles/s41467-025-61575-7)