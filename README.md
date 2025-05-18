# Deep Learning from Scratch

This project implements a full neural network framework and several models from scratch using Python and NumPy. It includes manual forward and backward propagation, training loops, and an autograder to verify correctness. The project covers Perceptron classification, regression, digit classification on MNIST, and language identification using RNNs.

## Features

- Neural network engine with computation graph
- Manual autograd system for backpropagation
- Parameter management and gradient updates
- Multiple models:
  - Perceptron (binary classification)
  - Regression (fit sin(x))
  - Digit Classification (MNIST)
  - Language Identification (RNN)
- Visual training interface with matplotlib
- Full autograding framework with feedback

## File Structure

| File | Description |
|------|-------------|
| `nn.py` | Core neural network primitives and layers |
| `models.py` | Model implementations (Perceptron, Regression, etc.) |
| `autograder.py` | Autograding logic and test cases |
| `backend.py` | Dataset generation and training visualizations |
| `VERSION` | Project version metadata |

## How to Use

### Run All Autograder Tests
```bash
python autograder.py
```
### Run Individual Model Tests

**Perceptron**
```bash
python autograder.py -q q1
```

**Regression**
```bash
python autograder.py -q q2
```

**Digit Classification**
```bash
python autograder.py -q q3
```

**Language Identification**
```bash
python autograder.py -q q4
```

### Optional Flags

- `--mute` to suppress training output
- `--no-graphics` to run without visualizations

## Requirements

- Python 3.x
- NumPy
- Matplotlib (optional, for graphics)
