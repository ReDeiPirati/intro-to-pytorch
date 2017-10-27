# FloydHub: Introduction to Deep Learning

## Introduction

Hi! Welcome to this mini-series on Deep Learning. Unless you're under complete isolation, you must have heard people talking about machine learning, deep learning & the infamous neural networks. Or some of you might have tried reading about Deep Learning, maybe going through some snippets, got frustrated with the overwhelming amount of technical jargon & wished someone would just give you a high-level explanation before you deep dived into the intricate details.

![AI-ML-DL](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/AI-ML-DL.png?raw=true)

This guide is for anyone who is curious about deep learning but has no idea where to start. Or someone who wishes to kickstart the their journey from low level introductory concepts to building high level state of the art deep learning models. The goal is to be accessible to anyone — which means that there are a lot of presumptions here as well. But who cares? If this gets you more interested in DL, then mission accomplished. Oh! And also, no advanced math required.

The terms "AI", "Deep Learning" & "Machine Learning" get thrown around so casually by developers, even the executives wish to implement AI in their services, but quite often, there isn't much clarity. Once you're through with this article you will be able to grasp the differences between important terms.

## Who should follow along?

- Developers who want to get up to speed on Deep Learning quickly.
- Non-technical folks who want a primer on Deep Learning and are willing to engage with technical concepts.
- Anyone curious to know how machine learn & think!

##

Table of Contents:
	- [Why AI matters?](#artificial-intelligence)
	- [ML](#machine-learning)
	- [DL](#deep-learning)
	- [DL](#deep-learning)
	- [Summary](#summary)

## Why AI matters?

A few phenomenal success stories like [DeepMind's AlphaGo](https://deepmind.com/blog/alphago-zero-learning-scratch/) & [Google's self driving car](https://waymo.com/) sparked the community's interest in this field & it has been uphill ever since. However, we don't see much of self driving cars around, do we? Or ever tried your hand at [Go](https://en.wikipedia.org/wiki/Go_(game))? Probably not. These landmark breakthroughs, however interesting they may seem at first, there is no denying that they're also a bit too abstruse to conceptually understand, and at the same time may make us overlook the humanly-fathomable & elegant solutions built using AI by developers sitting in their cubes all around the world. For instance,

1. Microsoft's [Seeing AI](https://www.microsoft.com/en-us/seeing-ai/) for the visually impaired beautifully integrates [object recognition](http://image-net.org/challenges/LSVRC/2017/) & [image captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/).
<a href="http://www.youtube.com/watch?feature=player_embedded&v=bqeQByqf_f8" target="blank"> <img style="float: center;" src="http://img.youtube.com/vi/bqeQByqf_f8/0.jpg" alt="Seeing AI" width="240" height="180" border="10"/></a>

2. [AI Experiments](https://experiments.withgoogle.com/ai) by Google is a collection of simple machine learning experiments ranging in areas from Computer Vision to Music Composition.
![AI_fields](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ai_exp.png?raw=true)

### What is Artificial Intelligence?

AI is nothing but an attempt to emulate human intelligence. Since its early days, research in AI  has been solely focussed on trying to replicate human intelligence for **specifc** tasks -- driving a car, playing chess etc.

![Machine_Brain](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ML_def.jpg)

### Is all this new?

To be honest, not really. We won't dive into the full history of AI, but it's good to know that the name AI was created by John McCarthy dedicated his life to the subject. To read more about the history of Deep Learning in general, you can check out the article, [Coding the History of Deep Learning](https://blog.floydhub.com/coding-the-history-of-deep-learning/) by Emil Walner.

## Machine Learning

Put simply, when a machine, by virtue of an algorithm, harnesses the ability to learn from a given set of data points without having to explicitly program the rules of the domain, it falls under the ambit of machine learning.  

### Trinity of Machine Learning

<img style="float: center;" src="https://image.slidesharecdn.com/nextgentalk022015-150211154330-conversion-gate02/95/an-introduction-to-supervised-machine-learning-and-pattern-classification-the-big-picture-8-638.jpg?cb=1423785060">

*Supervised Learning:* Say you get a bunch of photos with information on **what's on them** & you train a model to recognize new photos. Wanna see supervised learning in action? Head over to this example of [Teachable Machine](https://teachablemachine.withgoogle.com/) by Google.

*Unsupervised Learning:* Now you have a lot of molecules, some of them are drug-molecules & some are not but you **do not know which are which** and you want algorithm to discover the drugs, that is, you wish to segregate these molecules into 'clusters' of drug & non-drug molecules. You can watch [this video](https://youtu.be/wvsE8jm1GzE) here about Visualizing High Dimensional Space to see unsupervised learning in action.

*Reinforcement Learning:* Reinforcement helps map situations to actions with an end goal of maximizing the reward. Imagine a child trying to learn how to walk. The first thing he does, is observe how you walk. Grasping this concept, he/she tries to replicate you. But to walk, the child must first stand up! Staggering & slipping but still determined, clutching thin air to find support, the child finally manages to stand up & stay standing. Since there are so many things to keep in mind before actually walking, like balancing the body weight, deciding where to put your foot next etc. Now the child here is an agent trying to emulate walking by taking actions & he gets a reward (a satisfaction or a candy) when he completes a part of this task. This is a simplified analogy to any Reinforcement Learning problem. [DeepMind's AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) is one of the best examples of Deep Reinforcement Learning.

To get a brief intro to Machine Learning & some basic concepts, you can refer to [this article](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471) on Medium by Adam Geitgey.

## Machine Learning to Deep Learning

![DL_ML](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ML_DL.png?raw=true)

Deep learning is a sub domain of Machine Learning that involves algorithms 'inspired' by the human brain, to tackle machine learning problems. So what is a Neural Network really? Imagine if you were trying to solve a math equation to find the value of an unknown variable. Naturally, we simply cannot look at an equation & guess the answer! So what do we do? We try to solve parts of it, step by step. In each step, we try to simplify a part of this equation to work our way up to the solution. A neural network works in the same manner, where each step corresponds to a 'layer' in the network. Every layer is meant to solve a part of the problem. And much like math, if our answer doesn't match in the end, we trace back on those very steps to see what needs to be fixed! A deep neural network works in a similar manner, processing the input one step at a time. We'll dive into the details of 'training' a DNN in the upcoming articles.

![ml_to_dl](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ml_to_dl.png)

## How does FloydHub help?

Deep learning and its practise faces some practical difficulties. Most notably,
1. *Data is Primary, Learning is Secondary:* Datasets used to practise applications of DL like Computer Vision & NLP are huge. Take [this 'little' dataset](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data) from Kaggle. That's nearly 6.54 Gigabytes of training data. But all you wanted, was to train a simple Neural Network in Python. So how do you deal with that?
2. *AI wants compute, AI wants to multitask:* Noticed how the AI bigwigs always talk about GPUs & Tensors? Well GPUs are great at running parallel tasks and training DL models requires a lot of that compute. So what's the problem? Well the state of the art GPUs required for the purpose don't come in handy and the entire system built on top of them costs very good money. Take a look at [Andrej Karpathy's gig](https://twitter.com/karpathy/status/648256662554341377), one of the pioneers of DL. So, whether or not you should buy a GPU, is entirely up to you. If you do however need a perspective, you can refer to [this article](https://blog.floydhub.com/should-i-buy-my-own-gpus-for-deep-learning/).

FloydHub takes care of those two things for you! It manages the data & gives you lots of compute with a minimalistic interface. To setup your own first project on FloydHub, refer to the [QuickStart Doc](https://docs.floydhub.com/getstarted/quick_start/) or head over to [this tutorial](https://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub/).

## Summary

This was a high level introduction to the exciting field of Artificial Intelligence. Our hope is that this little introduction has inspired you to explore this domain. In the upcoming series of articles, we'll take you through a journey from introducing PyTorch, a great DL framework built in Python to implementing CNNs & training them on the cloud, with large enough datasets and state of the art compute. What do you need? Well, a working internet connection & a zeal to explore. We'll take care of the rest.

Next up in this series: [Introduction to PyTorch](https://github.com/ReDeiPirati/intro-to-pytorch/blob/sw/beginner/1.PyTorch/PyTorch_intro.md)
