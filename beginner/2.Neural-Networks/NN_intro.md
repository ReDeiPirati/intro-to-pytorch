# FloydHub Introduction to Deep Learning: Neural Networks

<p align="center">
  <img src="https://media.giphy.com/media/hq7TfiG7jxPTq/source.gif"/>
</p>

## Introduction

Welcome to the second part in this series. In this article, we'll introduce you to the `hello world` of deep learning-- The Handwritten Digit Recognition task. We'll (*try*) to building a simple & minimal Neural Network (from scratch) in PyTorch. We will also deal with some lingo revolving around Neural Networks such as a neuron, weights, biases etc. Without any delay, let's get started!

#### Quick Recap

Before we move ahead with our first task of implementing `Logistic Regression` for Digit Recognition, let's quickly mull over what we've done so far:

  1. We introduced PyTorch & some of its basic concepts like Tensors & Variables.
  2. A brief explanation about Autograd & Optimizers.
  3. Introduced the task of Handwritten Digit Recognition on the MNIST dataset.

## Logistic Regression

At the end of last lesson, we introduced the famous MNIST task. It is when we're given a `[28 x 28]` grayscale image of a handwritten digit, we wish to identify which one is it from `[0-9]`. The outputs here (digits) are referred to as classes, 10 of them in this case. This type of problem is known as **Multinomial Logistic Regression**-- when a response can have 3 or more values (10 in this case).

We'll now begin to understand the workflow of a classification problem. In the next few sections you'll see how an image dataset is processed, instantiation of a Logistic model in PyTorch & what more can we do!

For the sake of notation, let's denote the input image matrix `[28 x 28]` by `X` or `features` & the digit associated with that image as `Y` or `class label`.

#### *Step 1: Feature & Label Transformation*

We can convert the 2-dimensional image features in our example into a 1-dimensional one (as required in linear regression) by appending each row of pixels one after another to the end of the first row of pixels as shown below.

![Feature transformation](https://cdn-images-1.medium.com/max/1280/1*Vo8PMHppg_lWxFAZHZzNGQ.png)

Also, in a classification task, we cannot leave `Y` as a scalar value since we do not want the predicted outcome to be a real valued number.

To overcome this, the prediction from the model, `Y` should be transformed into a single column vector (shown below as row vector to conserve space) where each element represents the score of what the our classification model thinks is likely to be a particular class. In the example below, `class ‘1’` is the predicted class since it has the highest score.

![Predicted Outcome Transformation](https://cdn-images-1.medium.com/max/1280/1*Ld1fM5euVXm16mTf-4ifZA.png)

To derive this vector of scores, for a given image, each pixel on it will contribute to a set of scores (one for each class) indicating the *likelihood* by which it thinks the image is in a particular class. The sum of all the scores from every pixel for each class becomes the prediction vector.

![Predicted Outcome Transformation 2](https://cdn-images-1.medium.com/max/1280/1*aOP0s2i587kDJW2Td7GNqQ.png)

#### *Step 2: Cost Optimization*

If you remember, in the last article, we had defined our error/loss as `error = (y_-y).abs()` i.e the absolute numerical difference between the predicted output & the actual output. However, in a classification task, things are a bit different. Such a cost function, for an image of digit ‘1’, will penalize a prediction of ‘7’ more heavily `(7–1=6)` than a prediction of ‘2’ `(2–1=1)`, although both are equally wrong.

The most commonly used cost function in multi-class classification problem is the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss. It works in 3 steps,

  1. Convert actual image class vector `(y)` into a [one-hot vector](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/), which is a probability distribution.
  2. Convert prediction class vector `(y')` into a probability distribution using [sigmoid](https://www.quora.com/What-is-the-sigmoid-function-and-what-is-its-use-in-machine-learnings-neural-networks-How-about-the-sigmoid-derivative-function).
  3. Calculate the difference between 2 probability distribution functions.

To know more about the intricate functioning of Logistic Regression, read all about it in [this article](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54) by Soon Hin Khor.

### Handling the Data

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
```

`torch.utils.data.DataLoader` combines a dataset and a sampler, and provides single or multi-process iterators over the dataset.

```python
# Hyperparameter
batch_size = 64 #Number of image-label pairs processed at once.

# Training Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
# Testing Dataset Loader (Input Pipline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
```

### Build & Train

Now that we have our image-label pairs in order & ready to be processed in batches of 64 `(batch size)`, we can move ahead and define a standard logistic classifier with `CrossEntropy` loss.

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
        self.linear = nn.Linear(input_size, num_classes) # Define linear, y = x.W + b where W is weight matrix

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

And to iterate over the entire dataset to train on the above model architecture over 5 epochs,

```python
# Hyperparameters
num_epochs = 5
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
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        ### Keep track of metric for every every batch
        train_loss.append(loss.data[0])
        prediction = outputs.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(labels.data).sum()/batch_size*100
        train_accu.append(accuracy)
```

### Results: Logistic Regression

<p float="center">
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/loss_logreg.png?raw=true"/>
  <img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/accuracy_logreg.png?raw=true"/>
</p>

You can see here, how the decrease in the error/loss value leads to an increase in accuracy. This behaviour portrays the increasing confidence of the model as we feed in more & more data while iterating through the dataset. 
