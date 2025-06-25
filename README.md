# GraphDFT: Graph Neural Networks for Density Functional Approximation (Demo version)
GraphDFT is a PyTorch-based implementation of Graph Neural Networks (GNNs) for modeling exchange-correlation (XC) functionals in density functional theory (DFT).

This repository presents a demo version of the original codebase and is intended for showcasing model architecture and usage. Certain components such as proprietary data, full training scripts, or internal modules have been excluded for privacy and intellectual property reasons.

# Overview
Modern density functionals are limited by their local or semi-local nature. In this work, we use graph representations of molecular structures and train GNNs to learn functionals that go beyond traditional approximations. The goal is to provide:

- A flexible GNN architecture for XC potential modeling

- Integration with electronic structure data 

- A framework for developing machine-learned density functionals

# Features

- GNN models built using PyTorch Geometric

- Support for molecular graphs with atom-wise features

- Dataset loading and preprocessing pipeline

- Training scripts with configurable hyperparameters

- Modular code structure for extending new architectures

# Model Architecture
The core architecture uses Graph convolutional neural networks (GCN) and message passing layers to capture atomic interactions. The predicted outputs approximate the exchange-correlation energy as well as the potential using the automatic differentiation with PyTorch.

# Project Structure
GraphDFT/
├── dataset/             # (Excluded) Local training/test data not included in this public repo
├── models/              # GNN model definitions
├── train/               # Training functions
├── tests/               # Tests
├── utils/               # Helper functions
├── requirements.txt     # Dependencies
└── README.md

# Getting Started
## Installation
git clone git@github.com:soumitribedi/GraphDFT.git
cd GraphDFT
pip install -r requirements.txt

## Example Usage
python main.py

python main.py

# Dataset
This project assumes access to molecular datasets with:
- GEOM file containing Atomic positions
- rho_wf file containing electron density on a grid 
- vxc file containing XC potential on a grid
- gridwts file containing integration grid weights

# Note
Due to privacy concerns, raw data and full training datasets are excluded from this public repository.

# License
This project is licensed under the MIT License. See LICENSE for details.
