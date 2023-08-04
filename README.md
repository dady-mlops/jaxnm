# JAXNM
My experiments with JAX framework and normal mode approach.

## Intro
[**Phillip Lippe**](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html), University of Amsterdam:

_"...But why should you learn JAX, if there are already so many other deep learning frameworks like PyTorch and TensorFlow? The short answer: because it can be extremely fast. For instance, a small GoogleNet on CIFAR10 ... can be trained in JAX 3x faster than in PyTorch with a similar setup. ... JAX enables this speedup by compiling functions and numerical programs for accelerators (GPU/TPU) just in time, finding the optimal utilization of the hardware. Frameworks with dynamic computation graphs like PyTorch cannot achieve the same efficiency, since they cannot anticipate the next operations before the user calls them. For example, in an Inception block of GoogleNet, we apply multiple convolutional layers in parallel on the same input. JAX can optimize the execution of this layer by compiling the whole forward pass for the available accelerator and fusing operations where possible, reducing memory access and speeding up execution. In contrast, when calling the first convolutional layer in PyTorch, the framework does not know that multiple convolutions on the same feature map will follow. It sends each operation one by one to the GPU, and can only adapt the execution after seeing the next Python calls. Hence, JAX can make more efficient use of the GPU than, for instance, PyTorch.
However, everything comes with a price. In order to efficiently compile programs just-in-time in JAX, the functions need to be written with certain constraints. Firstly, the functions are not allowed to have side-effects, meaning that they are not allowed to affect any variable outside of their namespaces. For instance, in-place operations affect a variable even outside of the function. Moreover, stochastic operations such as <code>torch.rand(...)</code> change the global state of pseudo random number generators, which is not allowed in functional JAX (we will see later how JAX handles random number generation). Secondly, JAX compiles the functions based on anticipated shapes of all arrays/tensors in the function. This becomes problematic if the shapes or the program flow within the function depends on the values of the tensor. For instance, in the operation <code>y = x[x>3]</code>, the shape of <code>y</code> depends on how many values of <code>x</code> are greater than <code>3</code>. ... Still, in most common cases of training neural networks, it is straightforward to write functions within these constraints."_

## Installation

To install a CPU-only version of JAX, which might be useful for doing local development on a laptop, you can run:
```
pip install --upgrade "jax[cpu]"
```
You can use double-precision numbers by setting that in the configuration using <code>jax_enable_x64</code>
```
from jax.config import config
config.update("jax_enable_x64", True)
```
