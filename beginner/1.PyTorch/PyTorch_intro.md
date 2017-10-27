# FloydHub Introduction to Deep Learning: PyTorch

![FloydHub handles a PyTorch image](images/FloydTorch.png)

#### Abstract

[PyTorch](http://pytorch.org/) is one among the numerous [deep learning frameworks](https://www.kdnuggets.com/2017/02/python-deep-learning-frameworks-overview.html) which allows data scientists and AI practitioners to create amazing deep learning models. PyTorch has been gaining much [praise & popularity]((https://www.oreilly.com/ideas/why-ai-and-machine-learning-researchers-are-beginning-to-embrace-pytorch)) lately due to the high level of flexibility & its imperative programming flow.

## Introduction

The motivation behind this article is to give you a hands on experience with machine learning workflow with an example of logistic regression & introduce PyTorch, with its strengths and weaknesses that the framework provides. Before we begin, you should know that the PyTorch's [documentation](http://pytorch.org/docs/master/) and [tutorials](http://pytorch.org/tutorials/) are stored separately. And sometimes they may not converge due to the rapid speed of development and version changes. So feel free to investigate the [source code](https://github.com/pytorch/pytorch), if you feel so. [PyTorch Forums](https://discuss.pytorch.org/) are another great place to get your doubts cleared up. If you do however have any doubts/queries regarding our examples or in general, do let us know on the [FloydHub Forum](https://forum.floydhub.com/), we'll be happy to help.

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

Here's a list of modules we need in order to run this tutorial:

1. [torch.autograd](http://pytorch.org/docs/master/autograd.html) Provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions.
2. [torch.nn](http://pytorch.org/docs/master/nn.html) Package provides an easy and modular way to build and train simple or complex neural networks.
3. [torchvision](http://pytorch.org/docs/master/torchvision/index.html) consists of popular datasets, model architectures & common image transformations.
4. [NumPy](http://www.numpy.org/) is the fundamental package for scientific computing with Python.

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
```
You can find more of these in the `iPython Notebooks` that come along with this article. If you do feel comfortable in NumPy, this shouldn't be anything new.
However, unlike NumPy, PyTorch Tensors can utilize GPUs to accelerate their numeric computations & PyTorch makes it ridiculously easy to switch from GPU to CPU & vice versa.

*Note: It is interesting to know that PyTorch can serve as a full fledged replacement for NumPy, as Tensors & ndarrays can be used interchangeably. You can checkout the ipython notebook for an implementation.*

### Variables and AutoGrad

Tensors are an awesome part of the PyTorch. But what about [Gradients & Derivates](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)? Of course we can manually implement them, but thankfully, we don't have to. [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) exists. And to support it, a PyTorch Tensor must be first converted into a variable.

![autograd.Variable](http://pytorch.org/tutorials/_images/Variable.png)
*Credit: [PyTorch Variable docs](http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)*

Variables are wrappers above Tensors. With them, we can build our [computational graph](https://colah.github.io/posts/2015-08-Backprop/), and compute gradients automatically. Every variable instance has two attributes: `.data` that contains the initial tensor itself and `.grad` that contains gradients for the corresponding tensor.

When using `autograd`, the forward pass over your network will define a computational graph; nodes in the graph will be Tensors, and edges will be functions that produce the output Tensors. Backpropagating through this graph then allows you to easily compute gradients.

Here's a quick snippet on how we go about using Autograd & Variables:

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


## Logistic Regression

We are now moving on to a classical problem in Computer Vision: Handwritten digit recognition with Logistic Regression. Until now we've seen how to use Tensors (n-dimensional arrays) in PyTorch & compute their gradients with Autograd. The handwritten digit recognition is an example of a **classification** problem; given an image of a digit we can to classify it as either 0, 1, 2, 3...9. Each digit to be classified is known as a class.

![logreg](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/mnist_logreg.jpeg?raw=true)

So in simple terms: we'll be given a greyscale image (28 x 28) of some handwritten digit. We'll process this image to get a 28 x 28 matrix of real valued numbers, which we call **features** of this image. Our objective would be to **map a relationship between these features & the probability of a particular outcome**. If you are not familiar with this kind of a task, or wish to seek a quick intro to Logistic Regression, give [this article](https://medium.com/data-science-group-iitr/logistic-regression-simplified-9b4efe801389) a quick 5 minute read & you're good to go.

We can define a Logistic Regression model for our given problem in PyTorch with the following snippet,

```python
# Hyperparameters
input_size = 784 # 28 * 28
num_classes = 10
learning_rate = 1e-3

# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)
print (model)
```
>Output

```python
LogisticRegression (
  (linear): Linear (784 -> 10)
)
```

The model structure tells us how the computation 'transforms' the image feature vector of dim 784 to a vector of 10 real valued outputs representing probabilities.

### Dataset

For this task we will use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. We've already uploaded the entire [dataset on FloydHub](https://www.floydhub.com/redeipirati/datasets/mnist). You can access the data via the `input` path. Head over to the `ipython notebook` if you haven't already to check this one out.

To learn how datasets are managed, you can checkout the [dataset documentation](https://docs.floydhub.com/guides/create_and_upload_dataset/) or checkout this quick [tutorial](https://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub/).

#### Data Handling in PyTorch

![Tf data loaders pipeline](https://cdn-images-1.medium.com/max/1280/1*S00VU2HiEjNZ35zlj2kqfw.gif)
*Credit: [TF reading data docs](https://www.tensorflow.org/api_guides/python/reading_data)*

`torch.utils.data.DataLoader` combines a dataset and a sampler, and provides single or multi-process iterators over the dataset.

```python
# Training Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
```

To create your own custom data loader, inherit `torch.utils.data.Dataset` and amend some methods. For instance,

```python
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
train_dataset = ImagesDataset(
    df=train_df,
    transform=data_transforms
)

# initialize data loader with the required params
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=10,
                                           shuffle=True,
                                           num_workers=16)

# fetch the batch(call to `__getitem__` method)
for img, target in train_loader:
    pass
```
During the course of this mini series we'll play a lot with Dataset and DataLoader!

There is something however, that you should know. Image dimensions processed by PyTorch are different from TensorFlow. They are [batch_size x channels x height x width]. Applying `torchvision.transforms.ToTensor()` makes this transformation. Some other useful utils can be found in the [transforms package](http://pytorch.org/docs/master/torchvision/transforms.html).

And that's all for now. You're ready to head over to the `ipython notebook` attached with this article & try tinkering with some hyper-paramters yourselves. Do Tweet us [@FloydHub_](https://twitter.com/FloydHub_) if you get some interesting results yourself!

## Summary

PyTorch provides an amazing framework with an awesome community that can support us in our DL journey. We introduce introduce PyTorch with Logistic Regression & in the next article you'll some more traditional use cases of PyTorch; We'll be implementing a single layer Neural Network from scratch as well as creating some 'strange' networks to give you a good idea how Dynamic Compute graphs make PyTorch so powerful.

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
