<h1 align="center">
  World Models PyTorch Implementation 🏎️
</h1>

---

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat&logo=pytorch)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-green?style=flat&logo=openaigym)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)](LICENSE)

A clean, interactive PyTorch implementation of ["World Models"](https://worldmodels.github.io/) by David Ha and Jürgen Schmidhuber.

<p align="center">
  <img src="imgs/CarRacing.gif" width="500" alt="Car Racing Environment">
</p>

## Overview
---

This project implements the complete World Models architecture for the CarRacing-v3 environment from Gymnasium. World Models consist of three components:

1. **Vision (V)**: A Variational Autoencoder (VAE) that compresses raw images into latent representations
2. **Memory (M)**: A Mixed Density Network with LSTM (MDN-RNN) that predicts future states
3. **Controller (C)**: A simple neural network policy trained with CMA-ES



## Interactive Notebooks
---

The implementation is organized into interactive notebooks that explain each component:

| Notebook | Description |
|----------|-------------|
| [1-Rollouts.ipynb](1-Rollouts.ipynb) | Generating dataset from environment interactions |
| [2-Vision (VAE).ipynb](2-Vision%20(VAE).ipynb) | Training the Variational Autoencoder |
| [3-Memory (rnn-mdn).ipynb](3-Memory%20(rnn-mdn).ipynb) | Building the MDN-RNN predictive model |
| [4-Controller (C).ipynb](4-Controller%20(C).ipynb) | Evolutionary training of the controller |
| [5-Videos.ipynb](5-Videos.ipynb) | Generating videos of model performance |

## Features
---

- **Pure PyTorch** implementation with clean, commented code
- **Interactive Visualization** of latent space and model predictions
- **End-to-End Pipeline** from data collection to agent training
- **Pre-trained Models** included in `checkpoints/` directory
- **Modular Design** allowing for experimentation with architectures

## Visualizations

### VAE Latent exploration tools
<p align="center">
  <img src="imgs/latent_exploration.png" width="500" alt="Latent Space Visualization">
</p>

### Pygmae interactive game visualzation of trained models with keyboard controls
<p align="center">
  <img src="imgs/CarRacing_pygame.png" width="500" alt="Pygame interactive visualization">
</p>



@article{https://doi.org/10.5281/zenodo.1207631,
  doi = {10.5281/ZENODO.1207631},
  url = {https://zenodo.org/record/1207631},
  author = {Ha, David and Schmidhuber, Jürgen},
  title = {World Models},
  publisher = {Zenodo},
  year = {2018},
  copyright = {Creative Commons Attribution 4.0}
}

---
## License

This project is open-sourced under the [MIT License](LICENSE).
---