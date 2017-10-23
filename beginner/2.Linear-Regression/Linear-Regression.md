# FloyHub introduction to Deep Learning: Linear Regression
TODO: (More explicit about source transposition, with thanks and a good image.)

### Abstract
It’s Time to fire up the torch, we will cover the basic concept of ML to solve the hello world of ML: predict house estimation given square feet. Then we move on another classical: the handwritten digit recognition with Logistic Regression. *It’s ML time*.

## Introduction
We are going to solve an overly simple, and unrealistic problem, which has the upside of making understanding the concepts of ML and PyTorch easy. We want to predict a single scalar outcome, house price (in $) based on a single feature, house size (in square meters, sqm). This eradicates the need to handle multi-dimensional data, enabling us to focus solely on defining a model, implementing, and training it in PyTorch.

### Table of Contents

- ML Hello World
- Collect a Dataset
- Choose a Model (sub paraghaps)
- Train
- Evaluate
- What if we have more features?
- Do you want more?
- Summary
- It's you turn!
- One More Thing
- Thanks and Resources

## The ML Hello World

We start with a set of data points that we have collected (chart below), each representing the relationship between two values —an outcome (house price) and the influencing feature (house size).

![img - dataset](https://cdn-images-1.medium.com/max/1280/1*wcivD-w2dNHR7L3JUKwbhQ.png)

However, we cannot predict values for features that we don’t have data points for (chart below)

![img - how can we predict a new value given sqm](https://cdn-images-1.medium.com/max/1280/1*GH-vC3HDd01UFjjjCVMQlA.png)

We can use ML to discover the relationship (the ‘best-fit prediction line’ in the chart below), such that given a feature value that is not part of the data points, we can predict the outcome accurately (the intersection between the feature value and the prediction line.

![img - how can we predict a new value given sqm - visual solution](https://cdn-images-1.medium.com/max/1280/1*LMIk7UyRhz4ObI2FWX_75Q.png)

## Collect a Dataset

One of the most time and resource consuming task of every ML/DL workflow is to collect [high quality] dataset. What can we do? Well first of all you can Explore our datasets, otherwise you can search throught Internet.
Luckily Kaggle has provided a great dataset of sold houses in King County from May 2014 to 2015, we have already uploaded it for you so that you can immediatly start to play.

Let’s take a look about this dataset, but before we need to import the follow package:

- `torch` is our DL framwork
- `numpy` is the fundamental package for handling vector representation
- `pandas` simplify the dataset acquisition and exploration
- `matplotlib` to plot graphs


```python
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

cuda = torch.cuda.is_available()

# Seed for replicability
torch.manual_seed(1)
if cuda:
    torch.cuda.manual_seed(1)
```

```python
# The dataset is mounted in the /input path
data = pd.read_csv("/input/kc_house_data.csv")
# Return the first n=5 rows
data.head(n=5)
```
> Output

```python
# Describe our Dataset with some stats
data.describe()
```
> Output

Our main focus is trying to predict the price given the squarefeet. (You are free to make experiment on the other features)

```python
# Now let's plot Square Feet vs Price
# If you are European or want to reason in square meters: multiply sqft * 0.092903
plt.scatter(x=data.sqft_living, y=data.price/1e6)
plt.ylabel('Price in Milions of $', fontsize=12)
plt.xlabel('Square feet', fontsize=12)
plt.title("Price vs Square Feet")
```
> Output

```python
def dataset_normalization(ds):
    # Normalize data in [0,1] range
    # new_x = (x - min)/(max - min)
    min = ds.min()
    max = ds.max()
    return (ds - min)/(max - min)

data['price'] = dataset_normalization(data['price'])
data['sqft_living'] = dataset_normalization(data['sqft_living'])

# Let's Plot the Dataset after normalization
plt.scatter(x=data.sqft_living, y=data.price)
plt.ylabel('Price', fontsize=12)
plt.xlabel('Square feet', fontsize=12)
plt.title("Price vs Square Feet after Normalization in [0,1]")
```
> Output


```python
from torch.utils.data.dataset import Dataset

class Sqft2Price(Dataset):
    def __init__(self, start, end, normalize=False):
        self.data = pd.read_csv("/input/kc_house_data.csv")
        self.len = end
        if normalize:
            self.train = dataset_normalization(data['sqft_living'][start:end])
            self.label = dataset_normalization(data['price'][start:end])
        else:
            self.train = data['sqft_living'][start:end]
            self.label = data['price'][start:end]

    def __getitem__(self, index):
        # Return (train_sample, labels)
        return (self.train[index], self.label[index])

    def __len__(self):
        return self.len # how many examples you have

# Compute 20% for dev/test split
val_test_split = ((21613 * 20 )//100)
# The remaining 60 is for training
train_split = len(data.index) - (val_test_split * 2)

# Train Set Split %60
train_set = Sqft2Price(0, train_split, normalize=True)
print ("Train Set starts: 0, end: ", train_split)
# Val Set Split %20
val_split = train_split + val_test_split
print ("Validation Set starts: ", train_split+1, ", end: ", train_split + val_test_split)
val_set = Sqft2Price(train_split+1, train_split + val_test_split, normalize=True)
# Test Set Split %20
print ("Test Set starts: ", val_split+1, ", end: ", len(data.index))
test_set = Sqft2Price(val_split+1, len(data.index), normalize=True)
```
> Output

```python
# Hyper parameter
batch_size = 64

# Define the data Loader for Train, Val, Test
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

## Choose a Model

To do prediction using ML, we need to choose a model that can best-fit the data that we have collected.
We can choose a linear (straight line) model, and tweak it to match the data points by changing its steepness/gradient and position.

![img Linear Model visual](https://cdn-images-1.medium.com/max/1280/1*i8a-ADvmchTek5y9mWiImA.png)

We can also choose an exponential (curve) model, and tweak it to match the same set of data points by changing its curvature and position.

![img Curve Model visual](https://cdn-images-1.medium.com/max/1280/1*9aaM2_rUeMUkknRdoIDXFw.png)

Here's how to define a Linear Model in PyTorch.

```python
import torch.nn as nn

# Linear Regression Model
class LinearRegression(nn.Module):
    # Template for LR
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Weight Init
        nn.init.xavier_uniform(self.linear.weight)
        nn.init.constant(self.linear.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out

# Define a Linear Model with 1 feature(weight/variable) and bias, and output a single number
# w * sqft + bias = price, we are learning the w and bias
model = LinearRegression(input_size=1, output_size=1)

# If you are running a GPU instance, load the model on GPU
if cuda:
    model.cuda()
```

## Cost Function

To compare which model is a better-fit more rigorously, we define best-fit mathematically as a cost function that we need to minimize. An example of a cost function can simply be the absolute sum of the differences between the actual outcome represented by each data point, and the prediction of the outcome (the vertical projection of the actual outcome onto the best-fit line). Graphically the cost is depicted by the sum of the length of the blue lines in the chart below.

![img cost function visual](https://cdn-images-1.medium.com/max/1280/1*QaFrGv6YU357T97i5KcZgg.png)

**NOTE**: More accurately the cost function is often the squared of the difference between actual and predicted outcome, because the difference can sometimes can be negative; this is also known as min least-squared.

```python
# Use the Mean Squared Error Loss Function, MSE = (sum_over_n(pred - label)^2)/n where n are the number of samples
loss_fn = torch.nn.MSELoss(size_average=True)

# If you are running a GPU instance, compute the loss on GPU
if cuda:
    loss_fn.cuda()
```

## Gradient Descent

If you are on an expansive plateau in the mountains, when trying to descent to the lowest point, your viewpoint looks like this.

![mountain landscape](https://cdn-images-1.medium.com/max/1280/1*phKkGIPjF1_inKf46KF7EA.png)

The direction of descent is not obvious! The best way to descend is then to perform **gradient descent**:

- Determine the direction with the steepest downward gradient at current position
- Take a step of size X in that direction
- Repeat & rinse; this is known as training

Minimizing the cost function is similar because, the cost function is undulating like the mountains (chart below), and we are trying to find the minimum point, which we can similarly achieve through gradient descent.

![gsd](https://cdn-images-1.medium.com/max/1280/1*grA_mOjJddRb7kmvayMrWQ.png)

```python
# Define SGD Optimizer
learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

## Train

We train the model for one epoch, learning rate 1e-6 and SGD as optimizer.

```python
# Hyper Parameters
#input_size = 1
#output_size = 1
#learning_rate = 1e-6 # 0.000001
num_epochs = 1

model.train()
# Train the Model
for epoch in range(num_epochs):
    # Get batch from loader
    for i, (input, target) in enumerate(train_loader):
        # Transpose tensors
        input, target = input.view(-1,1), target.view(-1,1)
        float_tensor = 'torch.FloatTensor'
        # Convert to FloatTensor then wrap as Variable
        input = torch.autograd.Variable(input.type(float_tensor))
        target = torch.autograd.Variable(target.type(float_tensor))
        # Load Variable on GPU
        if cuda:
            input, target = input.cuda(), target.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Load loss into CPU
        if cuda:
            loss.cpu()
    if (epoch+1) % 1 == 0:
        print ('Epoch [%d/%d], Loss: %.4f'
               %(epoch+1, num_epochs, loss.data[0]))
```
> Output


## Evaluate

After the training of our model, we can take a look at the fitting line produced by our Linear Model: what is our expectation? Well we hope to see a strait line which "cut the dataset points in the middle".

```python
model.eval()
# Evaluate our Model on the full dataset
tensor = torch.from_numpy(data.sqft_living.values).type(float_tensor).view(-1,1)
tensor = torch.autograd.Variable(tensor)
if cuda:
    tensor = tensor.cuda()
predicted = model(tensor).cpu()
predicted = predicted.data.numpy()

# Plot the graph
plt.plot(data.sqft_living, data.price, 'bo', label='Original data')
plt.plot(data.sqft_living, predicted, 'r-', label='Fitted line')
plt.ylabel('Price', fontsize=12)
plt.xlabel('Square feet', fontsize=12)
plt.title("Linear Regression Model Learned")
plt.legend()
plt.show()
```
> Output

Excellent! But wait, we have not learned so much! Yes indeed, this is not a good model and it was obvius since the beginning that a line could not fit a similar distribution. We need deeper or powerful model capable to learn more than a single line.

## What if we have more features?

We have covered a linear model based on only the square feet living feature, but we know that the price is a function of different features, what do we need to do if we want to include additional features and try to predict the price with a Linear Model in this regression Task?

Now we have a new feature and we will see how to change the model according to this changing.

```python
from mpl_toolkits.mplot3d import Axes3D
fig1 =  plt.figure(figsize=(10, 10))
ax = fig1.gca(projection='3d')


# # Axes3D.scatter(xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True,)
data['price'] = dataset_normalization(data['price'])
data['sqft_living'] = dataset_normalization(data['sqft_living'])
data['bedrooms'] = dataset_normalization(data['bedrooms'])


ax.scatter(xs=data.sqft_living, ys=data.bedrooms, zs=data.price, c='b', marker='o', s=50)

ax.set_xlabel('Square feet')
ax.set_ylabel('Beedrooms')
ax.set_zlabel('Price')
plt.title("Price given Sqft and Beedrooms")
plt.show()
```
> Output

```python
class Sqft2Price(Dataset):
    def __init__(self, start, end, normalize=False):
        self.data = pd.read_csv("/input/kc_house_data.csv")
        self.len = end
        if normalize:
            self.train = dataset_normalization(data[['sqft_living', 'bedrooms']][start:end].values)
            self.label = dataset_normalization(data['price'][start:end].values)
        else:
            self.train = data[['sqft_living', 'bedrooms']][start:end]
            self.label = data[['sqft_living', 'bedrooms']][start:end]

    def __getitem__(self, index):
        # Return (train_sample, labels)
        return (self.train[index], self.label[index])

    def __len__(self):
        return self.len # how many examples you have

# Compute 20% for dev/test split
val_test_split = ((21613 * 20 )//100)
# The remaining 60 is for training
train_split = len(data.index) - (val_test_split * 2)

# Train Set Split %60
train_set = Sqft2Price(0, train_split, normalize=True)
print ("Train Set starts: 0, end: ", train_split)
# Val Set Split %20
val_split = train_split + val_test_split
print ("Validation Set starts: ", train_split+1, ", end: ", train_split + val_test_split)
val_set = Sqft2Price(train_split+1, train_split + val_test_split, normalize=True)
# Test Set Split %20
print ("Test Set starts: ", val_split+1, ", end: ", len(data.index))
test_set = Sqft2Price(val_split+1, len(data.index), normalize=True)
```
> Output

```python
# Batchsize
batch_size = 64

# Define the data Loader for Train, Val, Test
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

```python
# Linear Regression Model
class LinearRegression(nn.Module):
    # Template for LR
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Weight Init
        nn.init.xavier_uniform(self.linear.weight)
        nn.init.constant(self.linear.bias, 0)

    def forward(self, x):
        out = self.linear(x)
        return out

# Define a Linear Model with 2 features(weights/variables), bias, and output a single number
# w1 * sqft + w2 * bedrooms + bias = price, we are learning the w1, w2 and bias
model = LinearRegression(input_size=2, output_size=1)

# If you are running a GPU instance, load the model on GPU
if cuda:
    model.cuda()
```

```python
# Define SGD Optimizer
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

```python
# Hyper Parameters
#input_size = 1
#output_size = 1
#learning_rate = 1e-4 # 0.00001
num_epochs = 30

model.train()
# Train the Model
for epoch in range(num_epochs):
    # Get batch from loader
    for i, (input, target) in enumerate(train_loader):
        #print (input.size())
        float_tensor = 'torch.FloatTensor'
        # Convert to FloatTensor then wrap as Variable
        input = torch.autograd.Variable(input.type(float_tensor))
        target = torch.autograd.Variable(target.type(float_tensor))
        # Load Variable on GPU
        if cuda:
            input, target = input.cuda(), target.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Load loss into CPU
        if cuda:
            loss.cpu()
    if (epoch+1) % 1 == 0:
        print ('Epoch [%d/%d], Loss: %.4f'
               %(epoch+1, num_epochs, loss.data[0]))
```
> Output


```python
# TODO Update and Fix Graph Visualization
toshow = 100

# Normalization
# data['price'] = dataset_normalization(data['price'])
# data['sqft_living'] = dataset_normalization(data['sqft_living'])
# data['bedrooms'] = dataset_normalization(data['bedrooms'])

fig1 =  plt.figure(figsize=(10, 10))
ax = fig1.gca(projection='3d')

model.eval()
# Evaluate our Model on the full dataset
tensor = torch.from_numpy(data[['sqft_living', 'bedrooms']].values[:toshow]).type(float_tensor)
tensor = torch.autograd.Variable(tensor)
if cuda:
    tensor = tensor.cuda()
predicted = model(tensor).cpu()
predicted = predicted.data.numpy()

# Plot and Scatter
x, y = np.meshgrid(data.sqft_living.values[:toshow],data.bedrooms.values[:toshow])
ax.plot_surface(X=x, Y=y, Z=predicted, rstride=1, cstride=1, alpha=0.2, color='r')
ax.scatter(xs=data.sqft_living[:toshow], ys=data.bedrooms[:toshow], zs=data.price[:toshow], c='b', marker='o', s=50)

ax.set_xlabel('Square feet')
ax.set_ylabel('Beedrooms')
ax.set_zlabel('Price')
plt.title("Price given Sqft and Beedrooms")
plt.show()
```
> Output


## Do you want more?

In the next step we will move to another great classical of ML: the handwritten digit classifiaction task of MNIST dataset. We will cover another simple model (Logistic Regression) that we will "evolve"(Neural Networks) until reaching state-of-the-art model(Convolutional Neural Network) on this task and familiarize with PyTorch and the Deep Learning Framework.

## Summary

After this brief and high level introduction to Machine Learning and its workflow you have a developed a basic understanding about what ML is and in which way we can learn from data. But repeat everything one more time can only help us to fix this new knowledge, this is what we have learned:

- The First question every Data Scientist need to face is: Where Can I find the data usefull for my task?
- The next step is to create a data pipeline, in the previous example, we have defined a custom Dataset in which we have defined the train/dev/test split, then we have normalize our data for numerical stability.
- Once we have defined our dataset, we have build our Linear Model class inherited from the pytorch.nn module, according to the features and output we have to compute.
- A loss function and Optimizer are the last things to define before begin the training process
- In the training process we have wrapped all the pieces to learn a good model which can generalize on new data
- The evaluation process we are testing what our model have learned. Now we can make assumption and consideration abou the next experiment to run

## It's you turn!

Now it's your turn to test what you have learned on new knowledge: explore the dataset or take a new one, change hyperparamters, hack the linear model, explore new loss functions, try othera optimizera or if you are eager to learn, What are you expecting to move on the next part of the beginner episode? ;)

## One More Thing

You have achieved the *Linear Model Trophy*, now you are able to run the Polynomial Regression Task(it's a Linear Model capable to fit a 4degree Polynomial). How a Linear Model can fit a 4degree pol? Because it's Linear in the weights! It's ok if you are a bit confused. Take your time to fix this knowledge trying this example it will fix in your mind.

[Jump to PyTorch Polynomial Regression example](https://github.com/floydhub/regression)

## Thanks and Resources

Big thanks to:
- ...

Resources:
- [the-gentlest-introduction-to-tensorflow](https://medium.com/all-of-us-are-belong-to-machines/the-gentlest-introduction-to-tensorflow-248dc871a224)
- [yunjey-pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)
- [house-price-prediction, which have inspired me about use king county kaggle dataset](https://github.com/Shreyas3108/house-price-prediction/blob/master/housesales.ipynb)
- [pyTorch custom dataset example](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)