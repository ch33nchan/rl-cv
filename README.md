# TinyGrad RLCV

A lightweight reinforcement learning framework for computer vision tasks using TinyGrad.

## Overview

TinyGrad RLCV combines reinforcement learning with computer vision using the lightweight TinyGrad framework. It's designed to be efficient and deployable on resource-constrained devices like mobile CPUs and ARM-based systems.

## Features

- **TinyGrad Integration**: Efficient neural network operations with minimal memory footprint
- **Reinforcement Learning**: DQN implementation with experience replay and customizable policies
- **Computer Vision**: Lightweight image processing and feature extraction
- **Object Tracking**: Real-time tracking capabilities with webcam support
- **Performance Monitoring**: CPU, memory, and FPS tracking

## Requirements

- Python 3.10 or newer
- TinyGrad
- OpenCV (for webcam examples)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tinygrad-rlcv.git
cd tinygrad-rlcv
```
2. simple tracking (ps : work in progress):
```bash
cd tinygrad-rlcv/examples/
python webcam_tracking.py
```