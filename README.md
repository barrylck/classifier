# Digit Classifier Demo

**A digit classifier built from scratch using gradient descent**

## Problem
Classify handwritten digits (3 vs 7) from raw pixel values 
using a linear model trained with SGD.

## What This Demonstrates
- Images are tensors of floats — pixels mapped to 0.0–1.0
- A linear model is matrix multiplication + bias
- Loss functions must be smooth for gradients to flow
- The training loop: forward → loss → backward → update → zero_grad
- Sigmoid converts raw scores into probabilities (0 to 1)

## Architecture
Single linear layer: nn.Linear(784, 1)
- Input: 28×28 image flattened to 784 pixel values
- Output: one raw score per image
- Activation: sigmoid → converts to probability
- Loss: custom mnist_loss 

## Results
~98% validation accuracy after 5 epochs with lr=0.1

## How to Run
\```bash
pip install fastai
jupyter notebook classifier.ipynb
\```

## Key Concepts
See CONCEPTS.md for plain-English explanations of gradient 
descent, backpropagation, RAG vs fine-tuning, and more.

## Tech Stack
- Python 3.10
- fastai 2.x
- PyTorch
- Google Colab

## Key Concepts

**Gradient descent** — iteratively adjusts weights in the 
direction that reduces loss. Each step: forward pass → compute 
loss → backward pass → update weights → clear gradients.

**Loss vs accuracy** — loss is smooth and gradient-friendly 
(used for training). Accuracy is binary and human-readable 
(used for reporting). 

**Sigmoid** — squashes any raw score into 0.0–1.0 so it can 
be interpreted as a probability and compared to labels.