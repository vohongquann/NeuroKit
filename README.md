# Deep Learning Framework

A from-scratch deep learning framework with autograd system, inspired by PyTorch.

## Overview

This framework provides a complete deep learning ecosystem built from the ground up, featuring automatic differentiation, neural network modules, optimization algorithms, and data generation utilities. The implementation focuses on clarity and educational value while maintaining practical functionality for building and training neural networks.

## Core Components

### Tensor and Autograd

The `Tensor` class forms the foundation of the framework, supporting automatic differentiation through computational graphs. Each tensor maintains data, gradient information, and references to its parent tensors and gradient function.

Key features:
- Dynamic computational graph construction
- Gradient accumulation and backpropagation
- Operator overloading for intuitive syntax
- Support for both leaf and intermediate tensors

### Neural Network Modules

The `Module` base class provides the interface for all neural network components, with parameter management and training mode control.

Available layers:
- `Linear`: Fully connected layer with optional bias
- `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`: Activation functions
- `Softmax`: Multi-class classification output

### Loss Functions

- `MSELoss`: Mean Squared Error for regression tasks
- `CrossEntropyLoss`: Combined softmax and negative log likelihood for classification

### Optimizers

- `SGD`: Stochastic Gradient Descent with momentum and weight decay
- `Adam`: Adaptive Moment Estimation with bias correction
- `RMSprop`: Root Mean Square Propagation

### Data Generation

Multiple synthetic dataset generators for testing and demonstration:
- `Spiral`: Non-linearly separable spiral patterns
- `Line`: Linear classification data
- `Circle`: Concentric circular patterns
- `Zone`: Angular sector classification
- `Zone_3D`: 3D spherical distributions
- `GeneratePolynomialData`: Polynomial regression data

## Installation

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "neurokit"
version = "0.1.0"
description = "A lightweight neural network library."
authors = [{name = "Vo Hong Quan"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "jupyter",
    "matplotlib",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["neurokit*"]