# FloydHub Introduction to Deep Learning: Convolutional Neural Networks

<p align="center">
  <img src="https://blog.floydhub.com/static/5dbb96f16b91ac0639a42e2dfbd0d901-47668.jpg"/>
</p>

## Introduction

Hello there! Welcome to part 3 in FloydHub's introduction to Deep Learning mini-series. Before we begin, let's summarize over what we've been through so far,

  1. Introduced PyTorch - a popular deep learning framework based in Python, with some basic lingo associated with deep learning workflow.
  2. Explored how Neural Networks structurally function & how the data flows through the layered architecture.
  3. Implemented our first deep learning pipeline with the MNIST task of classifying handwritten digits through two methods -- The simple yet powerful logistic regression & a single layer Neural Network.

From this point on, we are going to explore the different Neural Network architectures designed in specific domains. For example, in this & the next article we'll be exploring the ever powerful Convolutional Neural Networks and their ranging applications in Computer Vision. We'll start off by implementing one on the MNIST data itself but gradually proceed to a more intuitive dataset. Our goal would be to make you understand (& implement) how this process of Convolution works & why it's far more efficient than deep multi-layer perceptrons.

It is interesting to know that by introducing ConvNets, we're finally moving to the stage in Deep Learning where the **GPU compute** we had talked about for so long, begins to shine. You'll see for yourself in a while.

## Motivation

Before understanding ConvNets, it is imperative to see why'd they came to be. We'll illustrate this using the same MNIST task. Remember how we used to plug in the pixels of an image & our network magically predicted the digit? Well, if you'd go through the notebook and see a sample of images in the MNIST dataset, there is one particular pattern you might notice -- the digit that the image represents **usually lies in the middle of the image**!

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/800/1*5ciREAL7xdyXcD-cSRP7Jw.png"/>
  </p>

So what happens when our test images do not follow this pattern? As you could probably guess...**it fails** to correctly classify the image. As another example, you could try feeding in a coloured image in our previous model, & it'll yet again fail because it has no idea what colours are or what their pixels represent. That doesn't seem very useful in the real world, now does it?

One particular solution to this problem is feeding in more data to the network; we apply a a ton of rotation, transposition, resizing operations on a single image to generate a higher number of samples of the same class & train a **deeper network**.

<p align="center">
<img src="https://cdn-images-1.medium.com/max/800/1*biD9eS5eB6zXzieonNk-VQ.png"/>
</p>

Now we have an endless supply of data & a very deep network to train with. So now, for every single image, we generate all its variants by altering color schemes, size ratios etc etc. Make much sense? Of course not!

And that's the motivation behind Convolutional Neural Networks! We wish to create a network architecture that is ***translationally invariant***.

## What's a ConvNet?

In layman terms, ConvNets look for symmetrical features in the input data. Each Convolutional layer has a set of **filters** that helps the network in this feature extraction. As the depth of such a network increases, complexity of the network increases as well. In our example, for instance, the first layer of convolution may capture simpler features such as the kind of channels (colors) while the last layer may capture complex features such as contours.

<p align="center">
<img src="http://machinelearninguru.com/_images/topics/computer_vision/basics/convolutional_layer_1/rgb.gif"/>
</p>

>Image: MLGuru

The illustration above portrays the operation of a convolutional layer. Imagine you have an RGB image, represented by `[number_of_channels (colors) x height x weight]`, in this case, `[3 x 5 x 5]` and we create `3` **filters** of `[3 x 3]` each. That is to say, `[3 x 3 x 3]` filters slide over the entire image and along the way take the dot products between filter values & chunks of the input image to get a `[1 x 4 x 4]` **feature map**. Since each convolutional layer consists of its own **filter**, if we stack `n` number of these with each independently convolved with the image, we end up with `n` feature maps! But **what's the point?** Because these filters are initialized randomly and thus, become our parameters which will be learned by the network subsequently.

Convolutional layers are often interweaved with **pooling layers**. Max Pooling, being a popular choice, takes the maximum of features over a small blocks of a previous layer. Thus, the output tells us **if a feature was present in the region, but not precisely where**.

<p align="center">
<img src="http://cs231n.github.io/assets/cnn/maxpool.jpeg"/>
</p>

How does pooling help? It allows the later convolutional layers to work on a larger subset of data, because a small patch of data *after pooling* represents a much bigger subsection before the pooling operation, making the network **invariant** to small transformations.

#### Striding & Padding

Striding & Padding are the two hyperparamters that can be used to alter the behaviour of each convolutional layer. **Stride** controls how the filter convolves of the image. The amount by which the filter shifts is the stride. In above illustration, the stride was implicitly set at 1. Stride is normally set in a way so that the output volume is an integer and not a fraction. Let’s look at an example.

<p align="center">
<img="https://adeshpande3.github.io/assets/Stride1.png"/>
</p>

A `[7 x 7]` input volume, with `[3 x 3]` filter with stride set to 1 gives a `[5 x 5]` feature map. Now let's try increasing the stride value to `2`,

<p align="center">
<img="https://adeshpande3.github.io/assets/Stride2.png"/>
</p>

The output volume shrinks! As you can see, we would normally increase the stride if we want less overlap & smaller spatial dimensions.

Now, let’s take a look at **padding**. Now imagine, what happens when you apply three 5 x 5 filters to a 32 x 32 x 3 input volume? The output volume would be a 28 x 28 x 3 feature map. Notice that the spatial dimensions decrease. As we make the network deeper, the size of the volume will decrease faster than we would like. In the early layers of our network, we want to preserve as much information about the original input volume so that we can extract those low level features. Let’s say we want to apply the same convolutional layer but we want the output volume to remain 32 x 32 x 3. To achieve this, we could apply a zero padding of size 2 to that layer. Zero padding pads the input volume with zeros around the border. If we think about a zero padding of two, then this would result in a 36 x 36 x 3 input volume. If you have a stride of 1 and if you set the size of zero padding to `(K - 1)/2` where K is the filter size, then the input and output volume will always have the same spatial dimensions.

<p align="center">
<img="https://adeshpande3.github.io/assets/Pad.png"/>
</p>

>The formula for calculating the output size for any given convolutional layer is `(W - K + 2P)/S + 1` where O is the output ratio of height/length, W is the input ratio of height/length, K is the filter size, P is the padding, and S is the stride.

### Hyperparameters

<p align="center">
<img src="https://media.mljar.com/blog/are-hyper-parameters-really-important-in-machine-learning/hyper-parameters.jpg"/>
</p>

Before moving on, let's take a minute to digest what we've learned so far and recapitulate. One particular aspect we'd like you think about here is about **Hyperparameters** -- things we encourage you to play with while trying out the `ipython notebook`.

  1. **ConvNet Layers:** How many? Filter Sizes? Stride/Padding? These may seem ordinary but they're not trivial matters. People often put in a lot of research into these factors since they determine the behaviour of convolutional operations.
  2. **ReLU Layers:** To introduce non-linearity (activation), it has been a convention to introduce one of these after every conv layer.  
  3. **Pooling Layers:** Max Pooling is surely the most popular choice here, but why? We leave this to you. Explore some other options & find out why!
  4. **1 x 1 convolution:** Often known as the `network-in-network` size filter. Performs N- Dimensional element-wise multiplications where N is the depth (color-channels) of the input volume into the layer.

There's sure more of these like Dropout, but more on that later.

## Building a ConvNet in PyTorch
