# FloydHub Introduction to Deep Learning: PyTorch

![FloydHub handles a PyTorch image](images/FloydTorch.png)

#### Abstract

[PyTorch](http://pytorch.org/) is one of the many [deep learning framework](https://www.kdnuggets.com/2017/02/python-deep-learning-frameworks-overview.html) which allows data scientists and AI practitioners to create amazing deep learning models. PyTorch has been gaining much [praise & popularity]((https://www.oreilly.com/ideas/why-ai-and-machine-learning-researchers-are-beginning-to-embrace-pytorch)) lately due to the high level of flexibility & its imperative programming flow.

## Introduction

The motivation behind this article is to give you a hands on experience with machine learning workflow with an example of logistic regression & introduce PyTorch, with its strengths and weakness that the framework provides. Before we begin, you should know that the PyTorch's [documentation](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/) are stored separately. And sometimes they may not converge due to the rapid speed of development and version changes. So feel free to investigate the [source code](https://github.com/pytorch/pytorch), if you feel so. [PyTorch Forums](https://discuss.pytorch.org/) are another great place to get your doubts cleared up. If you do however have any doubts/queries regarding our examples or in general, do let us know on the [FloydHub Forum](https://forum.floydhub.com/), we'll be happy to help.

**Table of Contents**:

- [PyTorch Introduction](#pytorch-introduction)
- [Tensors](#tensor)
- [Variables & Autograd](#variables-and-autograd)
- [Logistic Regression](#logistic-regression)
- [Summary](#summary)

### PyTorch Introduction

PyTorch is a Python based scientific computing package targeted at two sets of audiences:

- A replacement for NumPy to harness GPU compute capability.
- A Deep Learning research platform that provides maximum flexibility and speed.

*This introduction assume that you have a basic familiarity of NumPy, if it's not the case follow this [QuickStarter](https://cs231n.github.io/python-numpy-tutorial/#numpy) by Justin Johnson to get you up to speed.*

```python
# Import the package we need to run the tutorial
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import torchvision as tv

# Is cuda available on this instance?
cuda = torch.cuda.is_available()
```

### Tensor

First off, we introduce a fundamental concept of PyTorch concept: the Tensor. A PyTorch Tensor is conceptually identical to a numpy `ndarray`, and PyTorch provides many functions for operating on these Tensors. Like standard `ndarrays`, PyTorch Tensors do not know anything about deep learning or computational graphs or gradients; they are a generic tool for scientific computing.

The following snippets demonstrate Tensors & a few of their operations:

```python
# Construct a 5x3 matrix, uninitialized:
print("torch.Tensor(5, 3):")
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix
print("torch.rand(5, 3):")
x = torch.rand(5, 3)
print(x)

# There are multiple syntaxes for operations. Let’s see addition as an example
# Addition
print("Syntax 2: torch.add(x, y) =")
print(torch.add(x, y))

# Addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print("Syntax 3: torch.add(x, y, out=result) =")
print(result)

```
You can find more of these in the `iPython Notebooks` that come along with this article. If you do feel comfortable in NumPy, this shouldn't be new.
However, unlike NumPy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations & PyTorch makes it ridiculously easy to switch from GPU to CPU & vice versa.

Here we use PyTorch to convert a NumPy array to a PyTorch tensor and vice versa, then we will load a Tensor to GPU and then back again on CPU using cast before mentioned.

*Note: you can run the GPU to CPU example only if you are running a FloydHub GPU instance*

```python
# Generate a sample matrix of 3 rows and 4 columns
# from a Normal Distribution with Mean 0 and Var 1
numpy_tensor = np.random.randn(3, 4)
print ("Numpy tensor: ", numpy_tensor, "\n")

# Convert numpy array to pytorch array
pytorch_tensor = torch.Tensor(numpy_tensor)
print ("Numpy to PyTorch tensor: ", pytorch_tensor, "\n")

# If cuda is available, run GPU-to-CPU and vice versa example
if cuda:
    # If we want to use tensor on GPU provide another type
    dtype = torch.cuda.FloatTensor
    gpu_tensor = torch.randn(10, 20).type(dtype)
    # Or just call `cuda()` method
    gpu_tensor = pytorch_tensor.cuda()
    print ("PyTorch cuda gpu_tensor ", gpu_tensor, "\n")
    # Call back to the CPU
    cpu_tensor = gpu_tensor.cpu()
    print ("PyTorch cuda tensor to cpu_tensor, gpu_tensor.cpu() ", cpu_tensor, "\n")
# Define pytorch tensors
x = torch.randn(10, 20)
y = torch.ones(20, 5)
# `@` mean matrix multiplication from python3.5, PEP-0465
res = x @ y # Same as torch.matmul(x, y)
# Get the shape
res.shape  # torch.Size([10, 5])
```

### Variables and AutoGrad

Tensors are an awesome part of the PyTorch. But what about [Gradients & Derivates](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)? Of course we can manually implement them, but thankfully, we don't have to. [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) exists. And to support it, a PyTorch Tensor must be first converted into a variable.

![autograd.Variable](http://pytorch.org/tutorials/_images/Variable.png)
*Credit: [PyTorch Variable docs](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)*

Variables are wrappers above Tensors. With them, we can build our [computational graph](https://colah.github.io/posts/2015-08-Backprop/), and compute gradients automatically. Every variable instance has two attributes: `.data` that contains the initial tensor itself and `.grad` that contains gradients for the corresponding tensor.

When using `autograd`, the forward pass over your network will define a computational graph; nodes in the graph will be Tensors, and edges will be functions that produce the output Tensors. Backpropagating through this graph then allows you to easily compute gradients.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation that you can perform on a Tensor also works on Variables; the difference is that while using Variables, PyTorch defines a computational graph, allowing you to automatically compute gradients.

Here's a quick snippet on how we go about using Autograd & Variables:

```python
# Create a Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print("x", x)

# Do some operation
y = x + 2
print("x + 2 = y,", y, "\n")

# y was created as a result of an operation, so it has a grad_fn telling the history of its computation
print("y was created as a result of an operation, so we have ", y.grad_fn, "\n")

# More op
z = y * y * 3
out = z.mean()

print("y * y * * 3 = z", z, "\n", "mean(z), ", out)

# Let’s compute the gradient now
out.backward()
print("After backprop, x", x.grad)
```

## Logistic Regression


### Summary

PyTorch is an amazing framework from which starts a DL journey. Numpy extension, PyThonic, Flexible, easy to understand and debug. Already widely adopted by AI research community(FAIR is widely adopting it and NVIDIA is offering full support), unfortunately it’s not production ready as TF, it lack of a proper monitoring solution and it’s not widely available in term of blog/video resources as TF but we are sure that at the end of the Beta it will become mainstream.

PyTorch provides an amazing framwork with an awesome community that can support us in our DL journey.

If you have enjoied this Introduction, or you want to share your feedback(cheers, bug fix, typo and/or improvements), please leave a comment on our super active Forum webpage.

## Thanks and Resources

**Big thanks** to:
 - [Illarion Khlestov](https://medium.com/@illarionkhlestov) for the code snippets, image and article,
 - [PyTorch](http://pytorch.org/tutorials/) for the docs, code snippet, image and the amazing framework
 - [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) for the pytorch examples and snippet of code

Link References:
 - Pytorch [docs](http://pytorch.org/docs/master/) and [tutorial](http://pytorch.org/tutorials/)
 - [jcjohnson pytorch examples](https://github.com/jcjohnson/pytorch-examples)
 - [PyTorch tutorial distilled by Illarion Khlestov](https://medium.com/towards-data-science/pytorch-tutorial-distilled-95ce8781a89c)
