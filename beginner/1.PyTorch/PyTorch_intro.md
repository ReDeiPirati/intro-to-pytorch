# FloydHub Introduction to Deep Learning: PyTorch

<img style="float: center;" src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/PyTorch.jpg?raw=true">

## Introduction

[PyTorch](http://pytorch.org/) is one among the numerous [deep learning frameworks](https://www.kdnuggets.com/2017/02/python-deep-learning-frameworks-overview.html) which allows us to optimize mathematical equations using [Gradient Descent](https://medium.com/ai-society/hello-gradient-descent-ef74434bdfa5). The objective of this article is to give you a hands on experience with PyTorch & some basic mathematical operations that follow in the machine learning workflow. We also introduce the classic problem of [Handwritten Digit Recognition](http://yann.lecun.com/exdb/mnist/) a.k.a the hello world of Deep Learning.

*Follow the installation procedure from [here](http://pytorch.org/).*

**Table of Contents**:

- [PyTorch Introduction](#pytorch-introduction)
- [Tensors](#tensor)
- [Variables & Autograd](#variables-and-autograd)
- [Logistic Regression](#logistic-regression)
- [Summary](#summary)

### PyTorch Introduction

PyTorch is a Python based scientific computing package targeted at two sets of audiences:

- A replacement for NumPy to harness GPU compute capability.
- A Deep Learning research platform that provides maximum flexibility and speed through Dynamic Compute graphs & Imperative Programming control flow.

*This introduction assume that you have a basic familiarity with NumPy, if it's not the case follow this [QuickStarter](https://cs231n.github.io/python-numpy-tutorial/#numpy) by Justin Johnson and you're good to go.*

Here's a list of modules we'll need in order to run this tutorial:

1. [torch.autograd](http://pytorch.org/docs/master/autograd.html) Provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
2. [torch.nn](http://pytorch.org/docs/master/nn.html) Package provides an easy and modular way to build and train simple or complex neural networks.
3. [torchvision](http://pytorch.org/docs/master/torchvision/index.html) consists of popular datasets, model architectures & common image transformations.
4. [NumPy](http://www.numpy.org/) is the fundamental package for scientific computing with Python.

### Tensors

A [PyTorch Tensor](http://pytorch.org/docs/master/tensors.html) is conceptually identical to a numpy `ndarray`, and PyTorch provides many functions for operating on these Tensors. Like standard `ndarrays`, PyTorch Tensors do not know anything about deep learning or computational graphs or gradients; they are a generic tool for scientific computing. We can use n-dimensional Tensors to our requirement, for instance - we can have multidimensional (2D) tensor storing an image, or a single variable storing text.

The following snippets demonstrate Tensors & a few of their operations:

```python
dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# Construct a 5x3 matrix, uninitialized:
print("torch.Tensor(5, 3):")
x = torch.Tensor(5, 3).type(dtype)
print(x)

# Construct a randomly initialized matrix
print("torch.rand(5, 3):")
x = torch.rand(5, 3).type(dtype)
print(x)

# There are multiple syntaxes for operations. Let’s see addition as an example
# Addition
print("Syntax 2: torch.add(x, y) =")
print(torch.add(x, y))
```

Unlike NumPy `ndarrays`, PyTorch Tensors can utilize GPUs to accelerate their numeric computations & PyTorch makes it ridiculously easy to switch from GPU to CPU & vice versa. You can find more of these in the `iPython Notebooks` that come along with this article.

*Note: It is interesting to know that PyTorch can serve as a full fledged replacement for NumPy, as Tensors & ndarrays can be used interchangeably. You can checkout the ipython notebook for an implementation.*

### Variables and AutoGrad

![autograd.Variable](http://pytorch.org/tutorials/_images/Variable.png)
*Credit: [PyTorch Variable docs](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)*

Variables are **wrappers** over Tensors that can be differentiated & modified. [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) or `autograd` is a tool within PyTorch that helps us do just that. Every variable instance has two attributes: `.data` that contains the initial tensor itself and `.grad` that contains gradients for the corresponding tensor. Here's a quick snippet on how we go about using Autograd & Variables:

```python
# Create a Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
# Do some operation
y = x + 2
z = y * y * 3
out = z.mean()
# Let’s compute the gradient now
out.backward()
print("After backprop, x", x.grad)
```
>Output

```python
Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]
```

When we wrap our Tensors with Variables, the arithmetic operations still remain the same, but Variables also remember their history of computation. Thus, `z` is not only a regular `2 x 2` Tensor but expression, involving `y` & `x`. This is what helps us define a **computational graph**. Nodes in this graph are Tensors & edges will be Functions operated on these nodes. **Backpropagating** through this graph then allows you to easily compute gradients.

### Quick Trivia: Static vs Dynamic Compute Graphs

You may skip this section & will still do fine, but it's interesting to know how exactly TensorFlow & PyTorch differ & how is PyTorch so popular among Python developers.

![dynamic graph](http://pytorch.org/static/img/dynamic_graph.gif)

PyTorch's `autograd` looks a lot like TensorFlow: we define a computational graph, and use automatic differentiation to compute gradients. The difference between the two is that TensorFlow's compute graphs are **static** and PyTorch uses **dynamic** computational graphs.

In TensorFlow, we define the [computate graph](https://www.tensorflow.org/programmers_guide/graphs) once and then execute the same graph over and over again, like a loop. In PyTorch, each forward pass defines a new computational graph.

![TF data flow](https://www.tensorflow.org/images/tensors_flowing.gif)
*Credit: [TF Graph docs](https://www.tensorflow.org/programmers_guide/graphs)*

>Static graphs are nice because you can optimize the graph up front; framework might decide to fuse some graph
operations for efficiency, or to come up with a strategy for distributing the graph across many GPUs or many
machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over. However, for some models we may wish to perform
different computations differently for each data point; for example a recurrent network might be unrolled for different numbers of time steps for each data point; this unrolling can be implemented as a loop. With a static graph the loop construct needs to be a part of the graph; for this reason TensorFlow provides operators such as tf.scan for embedding loops into the graph. With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example, we can use normal imperative flow control to perform computation that differs for each input.

## Optimization

A quick recap of what we've learned so far:

  - What's a PyTorch Tensor & what does it represent.
  - What's a Variable & how to Tensors correspond to Variables.
  - How do we compute gradients & a brief intro into what exactly is a computational graph.

![GD](https://alykhantejani.github.io/images/gradient_descent_line_graph.gif)

The reason we wish to retain a computational graph of variables is so we can & update the variables to optimize equations. This may not make much sense now, but hang on for a while. We'll get there. Say we have two Variables `y_` & `y`. `y_` is what our model predicts & `y` is what it **should** predict (remember supervised learning?).

So how do we teach a machine that it's not doing it's job right of predicting `y` & needs to do better? We **optimize**.

```python
y = Variable(torch.FloatTensor([3.0]), requires_grad=True)
y_ = Variable(torch.FloatTensor([5.0]), requires_grad=True)

optimizer = torch.optim.SGD([x, y], lr=0.1)
for i in range(100):
    loss = (y_-y).abs()   # Minimizes absolute difference
    loss.backward()      # Computes derivatives automatically
    optimizer.step()     # Decreases loss
    optimizer.zero_grad()
print (y_)  # Evaluates to 3.0
print (y)  # Evaluates to 3.0 -- optimization successful
```

The above snippet creates an optimizer called Stochastic Gradient Descent, passing it a list of parameters to optimize & a [learning rate](https://medium.com/@balamuralim.1993/importance-of-learning-rate-in-machine-learning-920a323fcbfb). We try to minimize the difference between `y_` & `y`, slowly. And after 100 steps, they become equal.

We'll even use advanced optimizers like ADAGRAD & ADAM as well when we get to Neural Nets. They're usually slower & more explanatory but less likely to **overshoot** & thus, are used a lot. `torch.optim` module contains a number of these.

![optim](https://2.bp.blogspot.com/-eW63YjSyuwY/V1QP3b9ZSmI/AAAAAAAAFeY/VcLfkmRvGaQbRjKhetlKjIl59kgkGV6hQCKgB/s1600/opt1.gif)

## Next Up: Handwritten Digit Classification

We are now introducing a classical problem in Computer Vision: Handwritten Digit Recognition with Logistic Regression. Until now we've seen how to use Tensors (n-dimensional arrays) in PyTorch & compute their gradients with Autograd. The handwritten digit recognition is an example of a **classification** problem; given an image of a digit we can to classify it as either 0, 1, 2, 3...9. Each digit to be classified is known as a class.

<img style="float: center;" src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/mnist_logreg.jpeg?raw=true">

In simple terms: we'll be given a greyscale image (28 x 28) of some handwritten digit. We'll process this image to get a 28 x 28 matrix of real valued numbers, called **features** of this image. Our objective would be to **map a relationship between these features & the probability of a particular outcome**. Before moving on to the next article, if you are not familiar with this kind of a task, or wish to seek a quick intro to Logistic Regression, give [this article](https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389) a quick 5 minute read & you're good to go.

### Dataset

For this task we will use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. We've already uploaded the entire [dataset on FloydHub](https://www.floydhub.com/redeipirati/datasets/mnist) & you can access the same via the `input` path.

To learn how datasets are managed on FloydHub, you can checkout the [dataset documentation](https://docs.floydhub.com/guides/create_and_upload_dataset/) or checkout this quick [tutorial](https://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub/).

And that's all for now. You're ready to head over to the `ipython notebook` attached with this article & try some PyTorch exercises.

## Summary

PyTorch provides an amazing framework with an awesome community that can support us in our DL journey. We introduced PyTorch & in the next article you'll some more traditional use cases of PyTorch; We'll be implementing a full scale `Classification` exercise on PyTorch using Logistic Regression, look for some improvements through a single layer Neural Network as well as create some more 'strange' networks to give you a good idea how Dynamic Compute graphs make PyTorch so powerful.

*Note:* You should know that the PyTorch's [documentation](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/) are stored separately. And sometimes they may not converge due to the rapid speed of development and version changes. So feel free to investigate the [source code](https://github.com/pytorch/pytorch), if you feel so. [PyTorch Forums](https://discuss.pytorch.org/) are another great place to get your doubts cleared up. If you do however have any doubts/queries regarding our examples or in general, do let us know on the, we'll be happy to help.

We hope you enjoyed this Introduction to PyTorch. If you'd like to share your feedback (cheers, bug fix, typo and/or improvements), please leave us a comment on our super active [forum](https://forum.floydhub.com/) or tweet us [@FloydHub_](https://twitter.com/FloydHub_).

## Resources

**Big thanks** to:
 - [Illarion Khlestov](https://medium.com/@illarionkhlestov) for the code snippets, image and article,
 - [PyTorch](http://pytorch.org/tutorials/) for the docs, code snippet, image and the amazing framework
 - [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) for the pytorch examples and snippet of code

Link References:
 - Pytorch [docs](http://pytorch.org/docs/master/) and [tutorial](http://pytorch.org/tutorials/)
 - [jcjohnson pytorch examples](https://github.com/jcjohnson/pytorch-examples)
 - [PyTorch tutorial distilled by Illarion Khlestov](https://medium.com/towards-data-science/pytorch-tutorial-distilled-95ce8781a89c)
