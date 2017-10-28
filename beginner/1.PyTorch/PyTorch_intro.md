# FloydHub Introduction to Deep Learning: PyTorch

<img style="float: center;" src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/PyTorch.jpg?raw=true">

## Introduction

[PyTorch](http://pytorch.org/) is one among the numerous [Deep Learning frameworks](https://www.kdnuggets.com/2017/02/python-deep-learning-frameworks-overview.html) which allows us to build powerful Deep Learning models by harnessing GPU compute. PyTorch is extensively used for rapid prototyping in research and small scale projects. The objective of this article is to give you a hands on experience with PyTorch & some basic mathematical lingo associated with Deep Learning. We also introduce the classic problem of [Handwritten Digit Recognition](http://yann.lecun.com/exdb/mnist/).

**Table of Contents**:

- [PyTorch Introduction](#pytorch-introduction)
- [Tensors](#tensor)
- [Variables & Autograd](#variables-and-autograd)
- [Logistic Regression](#logistic-regression)
- [Summary](#summary)

### PyTorch Introduction

PyTorch is a Python based scientific computing package targeted at two sets of audiences:

- A Deep Learning research platform that provides maximum flexibility and speed through Dynamic Compute graphs & Imperative Programming control flow.
- A replacement for NumPy to harness GPU compute capability.


### Tensors

In any deep learning pipeline, one obvious inevitable thing that we encounter, is mathematical data. Be it an images stored in the form of `[height x width]` matrices, a piece of text stored in the form a vector or some spooky operation taking place between those two. PyTorch provides us with objects known as Tensors that store all this data under one roof.

*Formally*, a [PyTorch Tensor](http://pytorch.org/docs/master/tensors.html) is conceptually identical to a NumPy's `ndarray`, and PyTorch provides many functions for operating on these Tensors. Like standard `ndarrays`, PyTorch Tensors do not know anything about deep learning or computational graphs or gradients; they are a generic tool for scientific computing. We can use n-dimensional Tensors to our requirement, for instance - we can have multidimensional (2D) tensor storing an image, or a single variable storing text.

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

<p align="center">
  <img src="http://pytorch.org/tutorials/_images/Variable.png"/>
</p>

>Credits: PyTorch Variable Docs

Variables are **wrappers** over Tensors that allow them to be differentiated & modified. Let me demonstrate how: Take the example in the following snippet, where we apply a string of operations over a 'Variable' `x`, to predict `y`.

```python
import numpy as np
# x, w1, w2 is a PyTorch Variable
h = x.dot(w1)
h_relu = np.maximum(h,0)
y_pred = h_relu.dot(w2)

loss = np.mean(np.square(y_pred - y).sum())
```
And now, we wish to compute the derivative of this function with respect to the loss. Using the traditional symbolic differentiation, we would achieve that in a way like this,

```python
# Compute Gradient
grad_y_pred = 2.0*(y_pred - y)
grad_w2 = h_relu.T.dot(grad_y_pred)
grad_h_relu = grad_y_pred.dot(w2.T)
grad_h = grad_h_relu.copy()
grad_h[h < 0] = 0
grad_w1 = x.T.dot(grad_h)
```
>This process mentioned up is a part of Backpropagation in a simple single layer Neural Network, don't worry about it even if you don't understand much of it, we'll cover it in the next article.

Now imagine, if there were tens of different types of mathematical operations before computing `loss` in the first snippet (because there will be in what's about to come!). How could you possibly code the gradient computation for something like that? Thankfully, `torch.autograd` exists. It works on the principle of Automatic Differentiation, which is inherently based on the **chain rule**. To perform the gradient computation in the above example using `autograd`, all we have to do is,

```python
#Make sure x, w1 & w2 are Variables
y_pred = x.mm(w1).clamp(min=0).mm(w2)
loss = np.square(y_pred - y).sum()
loss.backward()
```

Every variable instance has two attributes: `.data` that contains the initial tensor itself and `.grad` that contains gradients for the corresponding tensor. Here are some more snippets on using Autograd & Variables:

```python
# Create a Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
# Do some operation
y = x + 2
z = y * y * 3
out = z.mean()
# Let’s compute the gradient now
out.backward()
print(x.grad)
```
>Output

```python
Variable containing:
 4.5000  4.5000
 4.5000  4.5000
[torch.FloatTensor of size 2x2]
```

*Note*: When we wrap our Tensors with Variables, the arithmetic operations still remain the same, but Variables also remember their history of computation. Thus, `z` is not only a regular `2 x 2` Tensor but expression, involving `y` & `x`. This is what helps us define a **computational graph**. Nodes in this graph are Tensors & edges will be Functions operated on these nodes. **Backpropagating** through this graph then allows you to easily compute gradients.

## Optimization

<p align="center">
    <img src="https://alykhantejani.github.io/images/gradient_descent_line_graph.gif"/>
</p>

So until now, we've seen Tensors that hold the data, Variables wrap around Tensors to let them perform complex math operations & finally `autograd` to compute gradients. So why do these Variables need to retain a history of computation?

The reason we wish to retain a computational graph of variables is so we can differentiate & update the variables to optimize mathematical equations. This may not make much sense now, but hang on for a while. We'll get there. Say we have two Variables `y_` & `y`. `y_` is what our model predicts & `y` is what it **should** predict (remember supervised learning?).

But how do we teach a machine that it's not doing a very good job of predicting `y` & needs to do better? You see, the basis of learning, be it biological beings like us or artificial machines, has always been 'repetition' of a  particular task i.e. a **learning algorithm**. To achieve this, we optimize!

```python
x = Variable(torch.FloatTensor([3.0]), requires_grad=False)
y_ = Variable(torch.FloatTensor([5.0]), requires_grad=True)
w = Variable(torch.randn(torch.FloatTensor(1)), requires_grad=True)

optimizer = torch.optim.SGD([y, y_], lr=0.1)
for i in range(100):
    error = (y_ - y*w).abs()   # Minimizes absolute difference
    error.backward()      # Computes derivatives automatically
    optimizer.step()     # Decreases loss: Updates y_ to become 'more' close to y
    optimizer.zero_grad()
print (y_)  # Evaluates to 3.0
print (y)  # Evaluates to 3.0 -- optimization successful
```

The above snippet creates an optimizer called Stochastic Gradient Descent, passing it a list of parameters to optimize & a [learning rate](https://medium.com/@balamuralim.1993/importance-of-learning-rate-in-machine-learning-920a323fcbfb). We try to minimize the difference between `y_` & `y`, slowly. And after 100 steps, they become equal.

We'll even use advanced optimizers like Adagrad & Adam when we get to Neural Nets. They're usually slower & more explanatory but are less likely to **overshoot** & thus, are used a lot. `torch.optim` module contains a number of these optimizers.

<p align="center">
    <img src="https://2.bp.blogspot.com/-eW63YjSyuwY/V1QP3b9ZSmI/AAAAAAAAFeY/VcLfkmRvGaQbRjKhetlKjIl59kgkGV6hQCKgB/s1600/opt1.gif"/>
</p>


### Quick Trivia: Why are we doing this in PyTorch? Why not TensorFlow?

You may skip this section & will still do fine, but it's interesting to know how exactly TensorFlow & PyTorch differ and how PyTorch is gaining so much popularity.

With PyTorch & Tensorflow, being the two most comprehensive & popular frameworks, it didn't take much time to boil down our options to these two. Even though TensorFlow is more popular, we chose to go ahead with PyTorch for two primary reasons.

1. **Graph Creation**: Creating & running graphs is where the two frameworks differ the most. Graphs in PyTorch are created dynamically, i.e at runtime. Whereas TensorFlow compiles the graph first, then executes it repeatedly. As a simple example, consider this:

```python
for _ in range(T):
    h = torch.matmul(W, h) + b
```
Since the above operation takes place under a standard Python loop, `T` can be changed with each iteration of this code. TensorFlow on the other hand uses its [control flow operations](https://www.tensorflow.org/api_guides/python/control_flow_ops#Control_Flow_Operations) making it a bit too tedious to compute a graph dynamically. Furthermore, this makes debugging much easier. You'll see some more virtues of dynamic compute graphs in the upcoming articles.

In TensorFlow, we define the [computate graph](https://www.tensorflow.org/programmers_guide/graphs) once and then execute the same graph over and over again, like a loop. In PyTorch, each forward pass defines a new computational graph.

<p align="center">
  <img src="https://www.tensorflow.org/images/tensors_flowing.gif"/>
</p>

>Credit: [TF Graph docs](https://www.tensorflow.org/programmers_guide/graphs)

>Static graphs are nice because you can optimize the graph up front; framework might decide to fuse some graph
operations for efficiency, or to come up with a strategy for distributing the graph across many GPUs or many
machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over. However, for some models we may wish to perform
different computations differently for each data point; for example a recurrent network might be unrolled for different numbers of time steps for each data point; this unrolling can be implemented as a loop. With a static graph the loop construct needs to be a part of the graph; for this reason TensorFlow provides operators such as tf.scan for embedding loops into the graph. With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example, we can use normal imperative flow control to perform computation that differs for each input.

2. **Data Loaders**: With its well designed APIs, sampler & data loader, parallelizing data-flow operations is incredibly simple. TensorFlow provides us with some of its data loading tools (readers, queues, etc) but PyTorch is clearly miles ahead.

*So why is TensorFlow so popular then?* While we may feel that learning about DL makes PyTorch a better candidate than TF, it may also be noted that there are certain fronts where TensorFlow does extremely well. Primarily in **Deployment**, **Device Management** & **Serialization**.


## Next Up: Handwritten Digit Classification

So that's all for now. For the next article in this series, we are introducing a classical problem in Computer Vision: Handwritten Digit Recognition. Until now we've seen how to use Tensors (n-dimensional arrays) in PyTorch & compute their gradients with Autograd. The handwritten digit recognition is an example of a **classification** problem; given an image of a digit we can to classify it as either 0, 1, 2, 3...9. Each digit to be classified is known as a class. We will (try) to build a classifier with only whatever you've learned until now & then finally introduce you to the Artificial Neural Networks.

<p align="center">
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/mnist_logreg.jpeg?raw=true"/>
</p>

Task: we'll be given a greyscale image (28 x 28) of some handwritten digit. We'll process this image to get a 28 x 28 matrix of real valued numbers, called **features** of this image. Our objective would be to **map a relationship between these features & the probability of a particular outcome**. Before moving on to the next article, if you are not familiar with this kind of a task, or wish to seek a quick intro to Logistic Regression, give [this article](https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389) a quick 5 minute read & you're good to go.

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
 - [Illarion Khlestov](https://medium.com/@illarionkhlestov) for the code snippets & images.
 - [PyTorch](http://pytorch.org/tutorials/) for the docs, code snippets, images and the amazing framework.
 - [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) for the pytorch examples and snippets of code.

Link References:
 - Pytorch [docs](http://pytorch.org/docs/master/) and [tutorial](http://pytorch.org/tutorials/)
 - [jcjohnson pytorch examples](https://github.com/jcjohnson/pytorch-examples)
 - [PyTorch tutorial distilled by Illarion Khlestov](https://medium.com/towards-data-science/pytorch-tutorial-distilled-95ce8781a89c)
