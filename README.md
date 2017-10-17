# FloydHub Introduction to Pytorch: a DL Framework

![FloydHub handles a PyTorch image](images/FloydTorch.png)

#### Abstract

[PyTorch](http://pytorch.org/) is an amazing [framework](https://en.wikipedia.org/wiki/Software_framework) which allows data scientists and AI practitioners to create amazing stuff. [Karpathy tweeted that this is the framework of 2017](https://twitter.com/karpathy/status/829518533772054529), [AI researchers are embracing it](https://www.oreilly.com/ideas/why-ai-and-machine-learning-researchers-are-beginning-to-embrace-pytorch) thanks to the high level of flexibility that the framework provide, moreover it’s pythonic!

## Introduction

This introduction want to explore the magic behind PyTorch, with the strengths and weakness that the framework provide. Before we start, you should know that the Pytorch [documentation](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/) are stored separately. Also sometimes they may don’t meet each other, because of fast development and version changes. So feel free to investigate [source code](https://github.com/pytorch/pytorch). It’s very clear and straightforward. It’s good to mention that there are exist awesome [PyTorch forums](https://discuss.pytorch.org/), where you may ask any appropriate question, and you will get an answer relatively fast. This place seems to be even more popular than StackOverflow for the PyTorch users.

**Table of Contents**:

- [Pytorch introduction](#pytorch-introduction)
- [Tensor](#tensor)
- [Variables & Autograd](#variables-and-autograd)
- [Defining new autograd functions](#defining-new-autograd-functions)
- [Static Vs Dynamic Computational Graph](#static-vs-dynamic-computational-graph)
- [Models Definition](#models-definition)
- [Train model with CUDA](#train-model-with-cuda)
- [Weight Init](#weight-initialization)
- [Excluding subgraphs from backward](#excluding-subgraphs-from-backward)
- [Training Process](#training-process)
- [Logging](#logging)
- [Data Handling](#data-handling)
- [Final architecture overview](#final-architecture-overview)
- [Summary](#summary)

*Note: During this introduction you will encounter ML/DL lingo and some training template with different models, even if you do not fully understand everything, don't worry, we will cover everything in a more concise way during the next episodes of this mini series.*

### Pytorch introduction

PyTorch is a Python based scientific computing package targeted at two sets of audiences:

- A replacement for numpy to use the power of GPUs
- A deep learning research platform that provides maximum flexibility and speed

*This introduction assume that you have a basic familiarity of numpy, if it's not the case, follow this [link](https://cs231n.github.io/python-numpy-tutorial/#numpy) to a well done Numpy Tutorial authored by [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)*

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

Numpy is a great framework, but it cannot utilize GPUs to accelerate its numerical computations. For modern deep neural networks, GPUs often provide speedups of [50x or greater](https://github.com/jcjohnson/cnn-benchmarks), so unfortunately numpy won't be enough for modern deep learning.

Here we introduce the most fundamental PyTorch concept: the Tensor. A PyTorch Tensor is conceptually identical to a numpy array: a Tensor is an n-dimensional array, and PyTorch provides many functions for operating on these Tensors. Like numpy arrays, PyTorch Tensors do not know anything about deep learning or computational graphs or gradients; they are a generic tool for scientific computing.

Here some example with PyTorch Tensor and some operations on them:

```python
# Construct a 5x3 matrix, uninitialized:
print("torch.Tensor(5, 3):")
x = torch.Tensor(5, 3)
print(x)

# Construct a randomly initialized matrix
print("torch.rand(5, 3):")
x = torch.rand(5, 3)
print(x)

# Get its size
print("Last Tensor Size:")
print(x.size())

# There are multiple syntaxes for operations. Let’s see addition as an example
# Addition: syntax 1
y = torch.rand(5, 3)
print("Syntax 1: x + y =")
print(x + y)

# Addition: syntax 2
print("Syntax 2: torch.add(x, y) =")
print(torch.add(x, y))

# Addition: giving an output tensor
result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print("Syntax 3: torch.add(x, y, out=result) =")
print(result)

# Addition: in-place
# adds x to y
print("In-place Addition: y.add_(x) =")
y.add_(x)
print(y)

# You can use standard numpy-like indexing with all bells and whistles!
print ("Indexing x[:, 1] - Second column(index starts from zero) of every rows:")
print(x[:, 1])
```

However unlike numpy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations. To run a PyTorch Tensor on GPU, you simply need to cast it to a new datatype, but also, it very simple to switch from GPU to CPU. Moreover it’s very easy to convert tensors from NumPy to PyTorch and vice versa.

Here we use PyTorch to convert a Numpy array to a PyTorch tensor and vice versa, then we will load a Tensor to GPU and then back again on CPU using cast before mentioned.

*Note: you can run the GPU to CPU example only if you are running a FloydHub GPU instance*

```python
# Generate a sample matrix of 3 rows and 4 columns
# from a Normal Distribution with Mean 0 and Var 1
numpy_tensor = np.random.randn(3, 4)
print ("Numpy tensor: ", numpy_tensor, "\n")

# Convert numpy array to pytorch array
pytorch_tensor = torch.Tensor(numpy_tensor)
print ("Numpy to PyTorch tensor: ", pytorch_tensor, "\n")
# Or another way
pytorch_tensor = torch.from_numpy(numpy_tensor)

# Convert torch tensor to numpy representation
print ("PyTorch to Numpy tensor: ", pytorch_tensor.numpy(), "\n")

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

Tensors are an awesome part of the PyTorch. But mainly all we want is to build some [neural networks](https://youtu.be/aircAruvnKk). What is about [backpropagation](https://www.quora.com/How-do-you-explain-back-propagation-algorithm-to-a-beginner-in-neural-network)? Of course, we can manually implement it, but what is the reason? Thankfully [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) exists. To support it PyTorch provides variables to you.

![autograd.Variable](http://pytorch.org/tutorials/_images/Variable.png)
*Credit: [PyTorch Variable docs](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)*

Variables are wrappers above tensors. With them, we can build our [computational graph](https://colah.github.io/posts/2015-08-Backprop/), and compute gradients automatically later on. Every variable instance has two attributes: `.data` that contain initial tensor itself and `.grad` that will contain gradients for the corresponding tensor.

When using autograd, the forward pass of your network will define a computational graph; nodes in the graph will be Tensors, and edges will be functions that produce output Tensors from input Tensors. Backpropagating through this graph then allows you to easily compute gradients.

PyTorch Variables have the same API as PyTorch Tensors: (almost) any operation that you can perform on a Tensor also works on Variables; the difference is that using Variables defines a computational graph, allowing you to automatically compute gradients.

Here we use PyTorch Variables and autograd: first on simple operations, second on forward and backward step for a linear model for a Regression Task with manual Gradient Descent update, and finally, we perform a some training steps using one hidden layer Neural Network with optimizer to show how to automatically update the weights during training.


```python
# Var and Authograd example on simple operations

# Create a Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
print("x", x)

# Make an op
y = x + 2
print("x + 2 = y,", y, "\n")

# y was created as a result of an operation, so it has a grad_fn.
print("y was created as a result of an operation, so we have ", y.grad_fn, "\n")

# More op
z = y * y * 3
out = z.mean()

print("y * y * * 3 = z", z, "\n", "mean(z), ", out)

# Let’s backprop now
out.backward()
print("After backprop, x", x.grad)
```

```python
# Var and Authograd example on single forward and backward step with manual GD(Gradient Descent)
# For reproducibility
torch.manual_seed(1)
# Define an dataset of 10 samples and 5 features
x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 1)
# Create Variable wrapper around Tensor
x = Variable(x_tensor, requires_grad=False)
y = Variable(y_tensor, requires_grad=False)
# Define some weights
w = Variable(torch.randn(5, 1), requires_grad=True)

# Get variable tensor
print("Dataset(sample) ", x_tensor, "\n", \
      "Dataset(labels), ", y_tensor, "\n", \
      "Type:", type(w.data), "\n")  # torch.FloatTensor
# Get variable gradient
print("At the beginnig w grad is ", w.grad, "\n")  # None

# MSE(Mean Squared Error Loss)
loss = torch.mean((y - x @ w) ** 2)

# Calculate the gradients
loss.backward()
print("After one forward step, w grad ", w.grad)  # some gradients
# Manually apply gradients - Gradient Descent Update
w.data -= 0.01 * w.grad.data
# Manually zero gradients after update
w.grad.data.zero_() # Tensor of 5 x 1 of zeros
```

```python
# Var and Authograd example on a single hidden layer NN with SGD training for few steps
# Remember: import torch.nn.functional as F

# For reproducibility
torch.manual_seed(1)

# Load your tensors on GPU if available
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Define an dataset of 10 samples and 10 features
x = Variable(torch.randn(10, 10).type(dtype), requires_grad=False)
y = Variable(torch.randn(10, 1).type(dtype), requires_grad=False)
# Define some weights
w1 = Variable(torch.randn(10, 5).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(5, 1).type(dtype), requires_grad=True)

# The lenght of the step we perform during GD
learning_rate = 0.1
# MSE(Mean Squared Error Loss)
loss_fn = torch.nn.MSELoss()
if cuda:
    loss_fn.cuda()
# Stocastic Gradient Descent Optimizer => w = w - lr * w.grads
optimizer = torch.optim.SGD([w1, w2], lr=learning_rate)
# Training Steps over full dataset
for step in range(5):
    # Hidden Layer
    hidden = F.sigmoid(x @ w1)
    # Model Output/ prediction
    pred = hidden @ w2

    # From Loss, Update the weight to improve prediction
    loss = loss_fn(pred, y)
    if cuda:
        loss = loss.cpu()
    l = np.asscalar(loss.data.numpy())
    print ("Loss {l} at step {i}".format(l=l, i=step))
    # Manually zero all previous gradients
    optimizer.zero_grad()
    # Calculate new gradients
    loss.backward()
    # Apply new gradients
    optimizer.step()
```

With the last example we have updated the weights automatically following these steps: Define an Optimizer, Compute the feedforward step, Zeroed the gradients Compute the Loss and BackProp(Compute the gradients with respect to Loss and Update the weights). But the main point that you should get from the last snippet: **we still should manually zero gradients before calculating new ones**. This is one of the core concepts of the PyTorch. Sometimes it may be not very obvious why we should do this, but on the other hand, we have full control over our gradients, when and how we want to apply them.

### Defining new autograd functions

Under the hood, each primitive autograd operator is really two functions that operate on Tensors. The *forward function* computes output Tensors from input Tensors. The *backward function* receives the gradient of the output Tensors with respect to some scalar value, and computes the gradient of the input Tensors with respect to that same scalar value.

In PyTorch we can easily define our own autograd operator by defining a subclass of torch.autograd.Function and implementing the forward and backward functions. We can then use our new autograd operator by constructing an instance and calling it like a function, passing Variables containing input data.

In this example we define our own custom autograd function for performing the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) nonlinearity, and use it with the code just above:

```python
# A one hidden layer NN with SGD training for few steps with ReLU(defined as autograd functions)

class MyReLU(torch.autograd.Function):
  """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """
  def forward(self, input):
    """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
    self.save_for_backward(input)
    return input.clamp(min=0)

  def backward(self, grad_output):
    """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
    input, = self.saved_tensors
    grad_input = grad_output.clone()
    grad_input[input < 0] = 0
    return grad_input


# For reproducibility
torch.manual_seed(1)

# Load your tensors on GPU if available
dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Define an dataset of 10 samples and 10 features
x = Variable(torch.randn(10, 10).type(dtype), requires_grad=False)
y = Variable(torch.randn(10, 1).type(dtype), requires_grad=False)
# Define some weights
w1 = Variable(torch.randn(10, 5).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(5, 1).type(dtype), requires_grad=True)

# The lenght of the step we perform during GD
learning_rate = 0.1
# MSE(Mean Squared Error Loss)
loss_fn = torch.nn.MSELoss()
if cuda:
    loss_fn.cuda()
# Stocastic Gradient Descent Optimizer => w = w - lr * w.grads
optimizer = torch.optim.SGD([w1, w2], lr=learning_rate)
# Training Steps over full dataset
for step in range(5):
    # Define our ReLU
    relu = MyReLU()
    # Hidden Layer
    hidden = relu(x @ w1)
    # Model Output/ prediction
    pred = hidden @ w2

    # From Loss, Update the weight to improve prediction
    loss = loss_fn(pred, y)
    if cuda:
        loss = loss.cpu()
    l = np.asscalar(loss.data.numpy())
    print ("Loss {l} at step {i}".format(l=l, i=step))
    # Manually zero all previous gradients
    optimizer.zero_grad()
    # Calculate new gradients
    loss.backward()
    # Apply new gradients
    optimizer.step()
```

### Static vs Dynamic Computational Graph

PyTorch autograd looks a lot like TensorFlow: in both frameworks we define a computational graph, and use automatic differentiation to compute gradients. The biggest difference between the two is that TensorFlow's computational graphs are **static** and PyTorch uses **dynamic** computational graphs.

In TensorFlow, we define the [computational graph](https://www.tensorflow.org/programmers_guide/graphs) once and then execute the same graph over and over again, possibly feeding different input data to the graph. In PyTorch, each forward pass defines a new computational graph. In the beginning, the distinction between those approaches not so huge. But dynamic graphs became very handful when you want to debug your code or define some conditional statements. You can use your favorite debugger as it is!

Static graphs are nice because you can optimize the graph up front; for example a framework might decide to fuse some graph operations for efficiency, or to come up with a strategy for distributing the graph across many GPUs or many machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over.

![TF data flow](https://www.tensorflow.org/images/tensors_flowing.gif)
*Credit: [TF Graph docs](https://www.tensorflow.org/programmers_guide/graphs)*

One aspect where static and dynamic graphs differ is control flow. For some models we may wish to perform different computation for each data point; for example a recurrent network might be unrolled for different numbers of time steps for each data point; this unrolling can be implemented as a loop. With a static graph the loop construct needs to be a part of the graph; for this reason TensorFlow provides operators such as tf.scan for embedding loops into the graph. With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example, we can use normal imperative flow control to perform computation that differs for each input.

Here's a comparison two definitions of the while loop statements - the first one in TensorFlow and the second one in PyTorch:

*Note: We have already provisioning this machine with Tensorflow, declaring this dependency in the `floyd_requirement.txt` file. So every time you need to use a package that is not available in the environment you will use, just remember to add that dependecy in the `floyd_requirement.txt` file. You can also run commands directly from Jupyter Notebook. For more infos about how to install extra dependecies, just take a look at our docs [here](https://docs.floydhub.com/guides/jobs/installing_dependencies/).*

```python
# Tensorflow Loop example
import tensorflow as tf

#### Constant and Variable ####
# Define the Variable and Constants we use in the computation
first_counter = tf.constant(0)
second_counter = tf.constant(10)
some_value = tf.Variable(15)

#### Computational Graph ####
# We build the CG defining the ops to perform on the Tensors
# Condition should handle all args:
def cond(first_counter, second_counter, *args):
    return first_counter < second_counter

# Add Ops
def body(first_counter, second_counter, some_value):
    first_counter = tf.add(first_counter, 2)
    second_counter = tf.add(second_counter, 1)
    return first_counter, second_counter, some_value

# Loop Op
c1, c2, val = tf.while_loop(
    cond, body, [first_counter, second_counter, some_value])

#### Session ####
# Where the execution takes place
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    counter_1_res, counter_2_res = sess.run([c1, c2])
    print (counter_1_res, counter_2_res) # 20, 20
```

```python
# PyTorch Loop example

first_counter = torch.Tensor([0])
second_counter = torch.Tensor([10])
some_value = torch.Tensor(15)

while (first_counter < second_counter)[0]:
    first_counter += 2
    second_counter += 1
print (first_counter, second_counter) # 20, 20
```

Code readability of PyTorch is really superior without any doubt.

### Models Definition

Computational graphs and autograd are a very powerful paradigm for defining complex operators and automatically taking derivatives; however for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation into **layers**, some of which have **learnable parameters** which will be optimized during learning.

In TensorFlow, packages like [Keras](https://github.com/fchollet/keras), [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim), and [TFLearn](http://tflearn.org/) provide higher-level abstractions over raw computational graphs that are useful for building neural networks.

In PyTorch, the `nn` package serves this same purpose. The `nn` package defines a set of **Modules**, which are roughly equivalent to neural network layers. A Module receives input Variables and computes output Variables, but may also hold internal state such as Variables containing learnable parameters. The `nn` package also defines a set of useful loss functions that are commonly used when training neural networks.

Do you remeber the one hidden layer NN the we used before? Now we will make the code `nn` compliant.

```python
# For reproducibility
torch.manual_seed(1)

# Define an dataset of 10 samples and 10 features
x = Variable(torch.randn(10, 10), requires_grad=False)
y = Variable(torch.randn(10, 1), requires_grad=False)

if cuda:
    x, y = x.cuda(), y.cuda()

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Variables for its weight and bias.
model = torch.nn.Sequential(
          torch.nn.Linear(10, 10),
          torch.nn.ReLU(),
          torch.nn.Linear(10, 1),
        )

if cuda:
    model.cuda()

# The lenght of the step we perform during GD
learning_rate = 0.1

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss()
if cuda:
    loss_fn.cuda()

# Stocastic Gradient Descent Optimizer => w = w - lr * w.grads
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Training Steps over full dataset
for step in range(5):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # From Loss, Update the weight to improve prediction
    loss = loss_fn(y_pred, y)
    if cuda:
        loss = loss.cpu()
    l = np.asscalar(loss.data.numpy())
    print ("Loss {l} at step {i}".format(l=l, i=step))
    # Manually zero all previous gradients
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    # Apply new gradients
    optimizer.step()
```

### Train model with CUDA

If was discussed earlier how we might pass one tensor to [CUDA](https://en.wikipedia.org/wiki/CUDA). But if we want to pass the whole model, it’s ok to call `.cuda()` method from the model itself, and wrap each input variable to the `.cuda()` and it will be enough. After all computations, we should get results back with `.cpu()` method. (Just retake a look at the code above).

```python
x = Variable(torch.randn(10, 10), requires_grad=False)
y = Variable(torch.randn(10, 1), requires_grad=False)

if cuda:
    x, y = x.cuda(), y.cuda() <== CUDA Variable
...

model = torch.nn.Sequential(
          torch.nn.Linear(10, 10),
          torch.nn.ReLU(),
          torch.nn.Linear(10, 1),
        )

if cuda:
    model.cuda() <== CUDA model

loss_fn = torch.nn.MSELoss()
if cuda:
    loss_fn.cuda() <= compute loss between cuda Tensor

...
# Inside Training

if cuda:
        loss = loss.cpu() <= From CUDA to CPU Tensor
...
```

### Weight initialization

In TensorFlow weights initialization mainly are made during tensor declaration. PyTorch offers another approach — at first, tensor should be declared, and on the next step weights for this tensor should be changed. Weights can be initialized as direct access to the tensor attribute, as a call to the bunch of methods inside `torch.nn.init` package. This decision can be not very straightforward, but it becomes useful when you want to initialize all layers of some type with same initialization.

Here's some examples:

```python
### 3 Ways to perform direct weights init ###

# New way with `init` module
w = torch.Tensor(3, 5)
torch.nn.init.normal(w)

# Work for Variables also
w2 = Variable(w)
torch.nn.init.normal(w2)

# Old styled direct access to tensors data attribute
w2.data.normal_()

### Weights Init for Module ###
# Example for some module
def weights_init(m):
    classname = m.__class__.__name__
    print (classname)
    if classname.find('Conv') != -1:
        print("Conv init")
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print("Batch init")
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# If you use this approach run the follow steps:
#
# model = Model()
# model.apply(weights_init)
#
# Follow the link for a full reference:
# https://github.com/floydhub/dcgan/blob/master/main.py#L96-L142


# For loop approach with direct access
class MyModel(torch.nn.Module):
    def __init__(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
# With the last approach the weights init will take place in the constructor
```

### Excluding subgraphs from backward

Sometimes when you want to retrain some layers of your model or prepare it for the production mode, it’s great when you can disable autograd mechanics for some layers. For this purposes, [PyTorch provides two flags](http://pytorch.org/docs/master/notes/autograd.html): `requires_grad` and `volatile`. First one will disable gradients for current layer, but child nodes still can calculate some. The second one will disable autograd for current layer and for all child nodes.

Here's an example:

```python
# Requires grad
# If there’s a single input to an operation that requires gradient,
# Its output will also require gradient.
x = Variable(torch.randn(5, 5))
y = Variable(torch.randn(5, 5))
z = Variable(torch.randn(5, 5), requires_grad=True)
a = x + y
print ("a.requires_grad", a.requires_grad)  # False
b = a + z
print ("b.requires_grad", b.requires_grad)  # True

# Volatile differs from requires_grad in how the flag propagates.
# If there’s even a single volatile input to an operation,
# Its output is also going to be volatile.
x = Variable(torch.randn(5, 5), requires_grad=True)
y = Variable(torch.randn(5, 5), volatile=True)
a = x + y
print ("a.requires_grad", a.requires_grad)  # False
```

### Training process

There are also some other bells and whistles in PyTorch. For example, you may use learning rate scheduler that will adjust your learning rate based on some rules. Or you may enable/disable batch norm layers and dropouts with single train flag. If you want it’s easy to change random seed separately for CPU and GPU.

Here's pseudo-code:
```python
# scheduler example
from torch.optim import lr_scheduler

# SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training template for 100 epochs
for epoch in range(100):
    scheduler.step()
    train()
    validate()

# Train flag can be updated with boolean
# to disable dropout and batch norm learning
# execute train step
model.train(True)
# or inside the train function just run
model.train()

# run inference step
model.train(False)
# or inside the validate function just run
model.eval()

# CPU seed
torch.manual_seed(42)
# GPU seed
torch.cuda.manual_seed_all(42)
```

Also, you may print info about your model, or save/load it with few lines of code. If your model was initialized with [OrderedDict](https://docs.python.org/3/library/collections.html) or class-based model string representation will contain names of the layers.

```python
model = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(1, 20, 5)),
    ('relu1', nn.ReLU()),
    ('conv2', nn.Conv2d(20, 64, 5)),
    ('relu2', nn.ReLU())
]))

print(model)

# Sequential (
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (relu1): ReLU ()
#   (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (relu2): ReLU ()
# )

save_path_params = 'model_params.ckp'

# save/load only the model parameters(prefered solution)
torch.save(model.state_dict(), save_path_params)
model.load_state_dict(torch.load(save_path_params))

save_path_model = 'model.ckp'
# save whole model
torch.save(model, save_path_model)
model = torch.load(save_path_model)
```

Now let's check the saved file with:

```bash
!ls model*
```

As per PyTorch documentation saving model with `state_dict()` method is [more preferable](http://pytorch.org/docs/master/notes/serialization.html).

*Note: If you want to load the model weights trained on GPU to CPU, use this: `torch.load('my_file.pt', map_location=lambda storage, loc: storage)` according to [this thread on PyTorch Discussion](https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349).*

### Logging

Logging of the training process is a pretty important part. Unfortunately, PyTorch has no any tools like [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). So you may use usual text logs with [Python logging module](https://docs.python.org/3/library/logging.html) or try some of the third party libraries:

- [A simple logger for experiments](https://github.com/oval-group/logger)
- [A language-agnostic interface to TensorBoard](https://github.com/torrvision/crayon)
- [Log TensorBoard events without touching TensorFlow](https://github.com/TeamHG-Memex/tensorboard_logger)
- [Tensorboard for pytorch](https://github.com/lanpa/tensorboard-pytorch)
- [Facebook visualization library wisdom](https://github.com/facebookresearch/visdom)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

During the next episodes we will try to explore these different solution to let you have a wide choice.

### Data handling

You may remember [data loaders proposed in TensorFlow](https://www.tensorflow.org/api_guides/python/reading_data) or even tried to implement some of them. This pipeline it's not very easy to understand, but it widely adopted.

![Tf data loaders pipeline](https://cdn-images-1.medium.com/max/1280/1*S00VU2HiEjNZ35zlj2kqfw.gif)
*Credit: [TF reading data docs](https://www.tensorflow.org/api_guides/python/reading_data)*

PyTorch developers decided do not reinvent the wheel(a classical [anti-pattern](https://en.wikipedia.org/wiki/Anti-pattern) in sw Engineering). They just use multiprocessing. To create your own custom data loader, it’s enough to inherit your class from `torch.utils.data.Dataset` and change some methods:

```python
# Remeber: import torchivision as tv

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None,
                 loader=tv.datasets.folder.default_loader):
        self.df = df
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        row = self.df.iloc[index]

        target = row['class_']
        path = row['path']
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        n, _ = self.df.shape
        return n

# what transformations should be done with our images
data_transforms = tv.transforms.Compose([
    tv.transforms.RandomCrop((64, 64), padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
])

train_df = pd.read_csv('path/to/some.csv')
# initialize our dataset at first
train_dataset = ImagesDataset(
    df=train_df,
    transform=data_transforms
)

# initialize data loader with required number of workers and other params
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=10,
                                           shuffle=True,
                                           num_workers=16)

# fetch the batch(call to `__getitem__` method)
for img, target in train_loader:
    pass
```

During the mini series we will have lot of fun and time for playing with Dataset and DataLoader :)

The two things you should know. First — image dimensions are different from TensorFlow. They are [batch_size x channels x height x width]. But this transformation can be made without you interaction by preprocessing step `torchvision.transforms.ToTensor()`. There are also a lot of useful utils in the [transforms package](http://pytorch.org/docs/master/torchvision/transforms.html).

The second important thing that you may use pinned memory on GPU. For this, you just need to place additional flag `async=Tru`e to a `cuda()` call and get pinned batches from DataLoader with flag `pin_memory=True`. More about this feature discussed [here](http://pytorch.org/docs/master/notes/cuda.html#use-pinned-memory-buffers).

### Final architecture overview

Now you know about models, optimizers and a lot of other stuff. What is the right way to merge all of them? I propose to split your models and all wrappers on such building blocks:

![Summary](https://cdn-images-1.medium.com/max/1280/1*A-cWYNur2lqDEhUF1_gdCw.png)

And here is pseudo-code template for a ML/DL script:

```python
class ImagesDataset(torch.utils.data.Dataset):
    pass

class Net(nn.Module):
    pass

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = torch.nn.MSELoss()

dataset = ImagesDataset(path_to_images)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

train = True
for epoch in range(epochs):
    if train:
        lr_scheduler.step()

    for inputs, labels in data_loader:
        inputs = Variable(to_gpu(inputs))
        labels = Variable(to_gpu(labels))

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if not train:
        save_best_model(epoch_validation_accuracy)
```

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
