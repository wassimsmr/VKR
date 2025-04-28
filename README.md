# VKR
# Neural Differential Equations Framework
# Overview
This project implements a modular, extensible framework for solving differential equations using both neural network-based approaches and classical numerical methods. It was specifically designed to compare the performance, accuracy, and efficiency of various solution methods across different types of differential equations.
# Key Features: 

- Multiple neural network architectures for solving differential equations:

1) Physics-Informed Neural Networks (PINN)
2) Standard Feedforward Neural Networks (MLP)
3) Recurrent Neural Networks (RNN)
4) Long Short-Term Memory Networks (LSTM)


- Classical numerical solvers:

1) Finite Difference Method (FDM)
2) Finite Element Method (FEM)


- Comprehensive metrics for comparison:

1) Solution accuracy (L2 error, L1 error, maximum error)
2) Computation/training time
3) Parameter count


- Visualization tools for analyzing results
- Support for different types of differential equations:

1) Linear ODEs
2) Nonlinear ODEs
3) Harmonic oscillators (second-order ODEs)



# Experiment Results
The experiments show that:

Neural network methods can achieve high accuracy for differential equations, comparable to traditional numerical methods.
For simple differential equations, classical methods like FDM typically solve problems faster, while neural methods require training.
For complex or high-dimensional problems, neural approaches can offer advantages in accuracy and generalization.
Among neural methods, PINNs generally provide the best balance of accuracy and computational efficiency.

# Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- Tqdm
