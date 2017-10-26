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

*Note: It is advised to follow along with the iPython Notebook while going through the entire article*

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

We are now moving on to a classical problem in Computer Vision: Handwritten digit recognition with Logistic Regression. Until now we've seen how to use Tensors (n-dimensional arrays) in PyTorch & compute their gradients with Autograd. The handwritten digit recognition is an example of a **classification** problem; given an image of a digit we can to classify it as either 0, 1, 2, 3...9. Each digit to be classified is known as a class.

## Logistic Regression vs Linear Regression

![LinR vs LogR](https://cdn-images-1.medium.com/max/1280/1*Oy6O6OdzTXbp_Czi_k4mRg.png)

*Differences*:

- **Outcome (y)**: For linear regression, this is a scalar value, e.g., $50K, $23.98K, etc. For logistic regression, this is an integer that refers to a class for e.g., 0, 1, 2, .. 9. Specifically, it is called regression because the output of each class is in range [0,1], but we have a classification task so in the end we have an output which represent one of this class.

- **Features (x)**: For linear regression, each feature is represented as an element in a column vector. For logistic regression involving a grayscale 2-D image, this is a 2-dimensional vector, with each element representing a pixel of the image; each pixel has a value of 0–255 representing a grayscale where 0 = black, and 255 = white, and other values some shade of grey.

- **Cost function (cost)**: For linear regression, this is some function calculating the aggregated difference between each prediction and its expected outcome. For logistic regression, this is some function calculating the aggregation of whether each prediction is right or wrong.

*Similarity*:

- **Training**: The training goals of both linear and logistic regression are to learn the weights (W) and biases (b) values.
- **Outcome**: The intention of both linear and logistic regression is to predict/classify the outcome (y) with the learned W, and b.

## Dataset

For this task we will use the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. We've already uploaded the entire [dataset on FloydHub](https://www.floydhub.com/redeipirati/datasets/mnist). You can access the data via the `input` path. Head over to the `ipython notebook` if you haven't already to check this one out.

To learn how datasets are managed, you can checkout the [dataset documentation](https://docs.floydhub.com/guides/create_and_upload_dataset/) or checkout this quick [tutorial](https://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub/).

The packages we'll be using are:

- `torch`, our DL framework
- `torchvision`, package to handle pytorch Dataset for computer vision task
- `torch.nn`, package we need to create our Models
- `numpy` package to handle vector representation
- `matplotlib` to plot graphs

The `torchvision` package consists of popular datasets, model architectures, and common image transformations for computer vision. `torchvision.datasets` provide a great API to handle the MNIST dataset. The snippet of code below, will create the MNIST dataset, then we will dive into to take a look about MNIST samples.

```python
# MNIST Dataset (Images and Labels)
# If you have not mounted the dataset, you can download it
# just adding download=True as parameter
train_dataset = dsets.MNIST(root='/input',
                            train=True,
                            transform=transforms.ToTensor())
x_train_mnist, y_train_mnist = train_dataset.train_data.type(torch.FloatTensor), train_dataset.train_labels
test_dataset = dsets.MNIST(root='/input',
                           train=False,
                           transform=transforms.ToTensor())
x_test_mnist, y_test_mnist = test_dataset.test_data.type(torch.FloatTensor), test_dataset.test_labels

print('Training Data Size: ' ,x_train_mnist.size(), '-', y_train_mnist.size())
print('Testing Data Size: ' ,x_test_mnist.size(), '-', y_test_mnist.size())

```

`torch.utils.data.DataLoader` combines a dataset and a sampler, and provides single or multi-process iterators over the dataset.

```python
# Hyperparameter
batch_size = 8

# Training Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# Testing Dataset Loader (Input Pipline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
```

## From Linear to Logistic Regression

To make logistic regression work with `y = W.x + b`, we need to make some changes to reconcile the differences stated above.

### Feature transformation, x

We can convert the 2-dimensional image features in our logistic regression example (assuming it has X rows, Y columns) into a 1-dimensional vector (as required in linear regression) by appending each row of pixels one after another to the end of the first row of pixels as shown below.

![Feature transformation](https://cdn-images-1.medium.com/max/1280/1*Vo8PMHppg_lWxFAZHZzNGQ.png)

### Predicted Outcome Transformation, y

For logistic regression, we cannot leave 'y' (predicted outcome) as a scalar since the prediction may end up being 2.3, or 11, which is NOT in the required set of classes [0, 1, …, 9].

To overcome this, the prediction 'y' should be transformed into a single column vector (shown below as row vector to preserve space) where each element represents the score of what the logistic regression model thinks is likely to be a particular class. In the example below, class ‘1’ is the prediction since it has the highest score.

![Predicted Outcome Transformation](https://cdn-images-1.medium.com/max/1280/1*Ld1fM5euVXm16mTf-4ifZA.png)

To derive this vector of scores, for a given image, each pixel on it will contribute a set of scores (one for each class) indicating the likelihood it thinks the image is in a particular class, based **ONLY** on its own greyscale value. The sum of all the scores from every pixel for each class becomes the prediction vector.

![Predicted Outcome Transformation 2](https://cdn-images-1.medium.com/max/1280/1*aOP0s2i587kDJW2Td7GNqQ.png)

### Cost Function Transformation

The cost function we are going to use is the cross entropy loss (H):

1. Convert actual image class vector (y’) into a one-hot vector, which is a probability distribution
2. Convert prediction class vector (y) into a probability distribution
3. Use cross entropy function to calculate cost, which is the difference between 2 probability distribution function

#### *Step 1. One-hot Vectors*

Since we have already transformed prediction (y) into a vector of scores, we should also transform the actual image class (y’) into a vector; each element in the column vector represents a class with every element being ‘0’ except the element corresponding to the actual class being ‘1’. This is known as a one-hot vector. Below we show the one-hot vector for each class from 0 to 9.

![one-hot-vector](https://cdn-images-1.medium.com/max/1280/1*YFh3GZ41PgQWAnB_LfmJVw.png)

Assuming the actual (y’) image being 1, thus having a one-hot vector of [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], and the prediction vector (y) of [1.3, 33, 2, 1.2, 3.2, 0.5, 3, 9.2, 1], plotting them for comparison becomes:

![one-hot vs prob](https://cdn-images-1.medium.com/max/1280/1*HtZU15da9Tip9kF83YY7Ag.png)

#### *Step 2. Probability Distribution with Softmax*

To mathematically compare similarity of two ‘graphs’, cross-entropy is a great way (and here is a fantastic albeit long explanation for those with a stomach for details).

To utilize cross entropy however, we need to convert both the actual outcome vector (y’) and the prediction outcome vector (y) values into a ‘probability distribution’, and by ‘probability distribution’ we mean:

- The probability/score of each class has to be between 0 to 1
- The sum of all the probabilities/score for all classes has to be 1

The actual outcome vector (y’) being one-hot vectors already satisfy these constraints.

For prediction outcome vector (y), we can transform it into a probability distribution using the Softmax function:

![softmax](https://cdn-images-1.medium.com/max/1280/1*gmOykUVXXUYK7LPDVZHMBg.png)

This is simply a 2-step process (see S1, S2 below), where each element in the prediction score vector (y), is exp’ed, and divided by the sum of the exp’ed total.

![how softmax works](https://cdn-images-1.medium.com/max/1280/1*yHfAuzJud7Za0BGOT-mwAg.png)

Note that Softmax(y) graph is similar in shape to the prediction (y) graph but merely with larger max and smaller min values.

![softmax vs pred](https://cdn-images-1.medium.com/max/1280/1*0BW3r-W89s9-8HUz_EhauA.png)

#### *Step 3. Cross Entropy*

We can now apply cross-entropy (H) between the predicted vector score probability distribution (y’) and the actual vector score probability distribution (y).

![CE](https://cdn-images-1.medium.com/max/1280/1*eNII64IH9v4JLJo-jfOBug.png)

To quickly understand this complex function, we break it down into 3 parts (see below). Note that, as notation in this article, we use y_i to represent “y with i subscript” in the formula H:

![splitting CE](https://cdn-images-1.medium.com/max/1280/1*MlJX2kL8vRPQU5xcgvS8bQ.png)

- Blue: Actual outcome vector, y_i’
- Red: -log of the probability distribution of prediction class vector, (Softmax(y_i)), explained previously
- Green: Sum of multiplication of blue and red components for each image class i, where i = 0, 1, 2, …, 9

The illustrations below should simplify your understanding further.

The blue plot is just the one-hot vector of actual image class (y’), see **One-hot Vector** section:

![one hot vect](https://cdn-images-1.medium.com/max/1280/1*-vPuYJvh8l7uSBKTl9Vm4w.png)

The red plot is derived from transformations of each prediction vector element, y, to Softmax(y), to -log(softmax(y):

![from pred to ce](https://cdn-images-1.medium.com/max/1280/1*AI9hbnU5SM8gLhGVhLASnQ.png)

The cross entropy (H), the green part (see below) is the multiplication of blue and red values for each class, and then summing them up as illustrated:

![CE explained](https://cdn-images-1.medium.com/max/1280/1*1wMxL3RsjfdSHhcbPd3R3g.png)

Since the blue plot is a one-hot vector, it only contains a single element of 1, which is for the correct image class, all other multiplications in the cross entropy (H) are 0, and H simplifies to:
```
Cross Entropy (H) = -log(softmax(y_i))
Where:
- y_i: Predicted score/probability for correct image class
```

## Building & Training

We have learned, in great detail, on how to transform a Linear Model to a Logistic one. In the next few snippets, we'll translate everything to PyTorch code. Follow along on the iPython Notebook.

```python
# Hyperparameters
input_size = 784 # 28 * 28
num_classes = 10
learning_rate = 1e-3

#### Model ####
# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

model = LogisticRegression(input_size, num_classes)

# If you are running a GPU instance, load the model on GPU
if cuda:
    model.cuda()


#### Loss and Optimizer ####
# Softmax is internally computed.
loss_fn = nn.CrossEntropyLoss()
# If you are running a GPU instance, compute the loss on GPU
if cuda:
    loss_fn.cuda()

# Set parameters to be updated.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Let's train the model for 5 epochs.

```python
# Hyperparameters
num_epochs = 5
print_every = 100

# Metrics
train_loss = []
train_accu = []

# Model train mode
model.train()
# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # image unrolling
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Load loss on CPU
        if cuda:
            loss.cpu()
        loss.backward()
        optimizer.step()

        ### Keep track of metric every batch
        # Loss Metric
        train_loss.append(loss.data[0])
        # Accuracy Metric
        prediction = outputs.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(labels.data).sum()/batch_size*100
        train_accu.append(accuracy)

        # Log
        if (i+1) % print_every == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f'
                   % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], accuracy))
```

And to evaluate on the Test Set,

```python
model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data.view(-1, 28*28), volatile=True), Variable(target)
    if cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    # Load output on CPU
    if cuda:
        output.cpu()
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))
```

To see the complete results & some samples from the dataset, head over to `ipython notebook` attached with this article & try tinkering with some hyperparamters yourselves. Do Tweet us [@FloydHub_](https://twitter.com/FloydHub_) if you get some interesting results yourself!

### Summary

PyTorch provides an amazing framework with an awesome community that can support us in our DL journey.
We hope you enjoyed this Introduction to PyTorch. If you'd like to share your feedback (cheers, bug fix, typo and/or improvements), please leave a comment on our super active Forum webpage or Tweet us.

## Thanks and Resources

**Big thanks** to:
 - [Illarion Khlestov](https://medium.com/@illarionkhlestov) for the code snippets, image and article,
 - [PyTorch](http://pytorch.org/tutorials/) for the docs, code snippet, image and the amazing framework
 - [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) for the pytorch examples and snippet of code

Link References:
 - Pytorch [docs](http://pytorch.org/docs/master/) and [tutorial](http://pytorch.org/tutorials/)
 - [jcjohnson pytorch examples](https://github.com/jcjohnson/pytorch-examples)
 - [PyTorch tutorial distilled by Illarion Khlestov](https://medium.com/towards-data-science/pytorch-tutorial-distilled-95ce8781a89c)
