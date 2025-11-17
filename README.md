# Edge Predictor with Diffusion Model for DNA Binding Sites Prediction

A deep learning framework combining edge prediction and diffusion models for accurate DNA-protein binding site prediction.

## 🎯 Overview

This project implements an advanced graph neural network approach for predicting DNA binding sites in protein sequences. The model leverages:
- **Edge Predictor**: Captures structural relationships between amino acids
- **Diffusion Model**: Generates high-quality augmented training data
- **GAT-GNN**: Graph Attention Networks for binding site classification

## 🚀 Features

- DNA-binding site prediction for protein sequences
- Data augmentation using diffusion models
- Cross-validation and ensemble learning
- Support for multiple DNA-binding protein families (HTH, zinc finger, bHLH, homeodomain)
- PyMOL visualization scripts for 3D structure analysis

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install scikit-learn pandas numpy
pip install biopython

## 📊 Model Architecture
Data Augmentation: Diffusion model generates synthetic protein graphs
Edge Prediction: Predicts structural edges between residues
Graph Construction: Builds protein graphs with predicted edges
GAT-GNN Classification: Multi-head attention for binding site prediction
Ensemble Learning: Combines multiple models for robust predictions
## 🧪 Supported Protein Types
p53 tumor suppressor protein
Lac Repressor (HTH domain)
MyoD (bHLH domain)
Homeobox proteins
Other DNA-binding proteins

