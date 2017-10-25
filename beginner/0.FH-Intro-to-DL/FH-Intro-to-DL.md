# FloydHub Introduction to Deep Learning

### Abstract

[(Prof) Andrew Ng, Coursera co-founder & Stanford Adjunct Professor, recently said that AI is the new electricity](https://youtu.be/21EiKfQYZXc), [even Google shifted its focus from mobile-first to AI-first](https://youtu.be/Y2VF8tmLFHw?t=6m50s), even the VCs with their startups are riding this wave of hype. Some even think of it as magic while others are really scared, primarily due to the premature sci-fi literature. But what is it really? With [Zuck vs Elon declarations](https://www.theguardian.com/technology/2017/jul/25/elon-musk-mark-zuckerberg-artificial-intelligence-facebook-tesla) on twitter and the narrative about human job replacement due to AI, we are often overwhelmed to an extent that we overlook the countless applications of AI that make ourselves so much more productive and motivates some of the brightest minds around the world to work in this area.

At FloydHub we firmly believe that, if widely understood and tested, this technology has the potential to change human-machine interaction providing additional value to the society. One of our fundamental goal is to `#makeAIsimple`. We want to build a platform providing dreamers & developers a toolbox to create, learn, and experiment & see *what change can they bring about.*. Shall we begin? ;)

Table of Contents:
	- [AI](#artificial-intelligence)
	- [ML](#machine-learning)
	- [DL](#deep-learning)
	- [DL](#deep-learning)
	- [Summary](#summary)

## Artificial Intelligence

Everyone likes to talk about the miraculous effectiveness of AI: AI can "see", AI can "understand" and AI can "whatsoever-a-people-can-do". Is this all true? Well if you are riding the the wave of hype created around the latest breakthroughs such as [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) or [Google's self driving car](https://waymo.com/), all of it may seem too abstruse, and at the same time make us overlook the humanly-fathomable & elegant solutions built using AI by developers sitting in their rooms all around the world. For instance,

1. Microsoft's [Seeing AI](https://www.microsoft.com/en-us/seeing-ai/) for the visually impaired beautifully integrates [object recognition](http://image-net.org/challenges/LSVRC/2017/) & [image captioning](http://cs.stanford.edu/people/karpathy/deepimagesent/).
<a href="http://www.youtube.com/watch?feature=player_embedded&v=bqeQByqf_f8" target="blank"> <img style="float: center;" src="http://img.youtube.com/vi/bqeQByqf_f8/0.jpg" alt="Seeing AI" width="240" height="180" border="10"/></a>

2. [AI Experiments](https://experiments.withgoogle.com/ai) by Google is a collection of simple machine learning experiments ranging in areas from Computer Vision to Music Composition.
![AI_fields](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ai_exp.png?raw=true)

### What is AI?

According to Professor P. Norvig book: Artificial Intelligence a modern approach, we can define AI in four ways: Agent which thinks like a human, Agent which acts like a human, Agent which thinks rationally and Agent which acts rationally. Well, this may sound a little weird but are we a rational species? It all depends on what we deem as rational, but for simplicity we are assuming that rationality means: reasoning, even with emotions involved, about planning or taking decisions so that the situation turns out to be "good" for the Agent.

### Why study AI?
Have you ever asked: How my brain works? Why can I do these thing? What's the difference from me to other animals? If you have asked this questions, do not feel you alone, otherwise: blissful carefreeness!

[Even if our behaviour is the outcome of 1ms, 1s, 1h, 24h, 1 month, 1y, 100y, 1000y, 10...0y of evolution](https://youtu.be/NNnIGh9g6fA), what is the algorithm of intelligence behind it? If we replace the term human with agent: which is the algorithm that drives us in an environment where we can act and receive stimulus/observations to maximize future rewards according to the goals defined by the algorithm? And are these rules hard coded? Or changing?
![Machine_Brain](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ML_def.jpg)

Human species is one of the most incredibly things ever created in term of complexity and this is the fundamentally the reason behind the successes in this field. Solving the "intelligence mystery" means unboxing human complexity and finally understanding how we do, what we do!

### Fields in AI

![AI_fields](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/AI.png?raw=true)

AI is an extremely multi-disciplinary field, it covers: Planning, Robotics, Math, Neuroscience, Computational Biology, Data Science, Natural Language Processing and Understanding, Knowledge Representation, Reasoning, and Machine Learning. Don't get overwhelmed already if you don't have a solid understanding of these. You'll learn a lot during this journey.

With the last achievement in this field, you can try to improve all the existing technologies and maybe simplify the human-machine interaction building a future where human possibilities will be so wide that the only limit is our imagination.

### How old is AI?

Pretty old. Even if we have not discussed the full history of AI, it's good to know that the name AI was created by John McCarthy who spent all his life on it. To read more about the history of Deep Learning in general, you can check out the article, [Coding the History of Deep Learning](https://blog.floydhub.com/coding-the-history-of-deep-learning/) by Emil Walner.

## Machine Learning

Put simply, when a machine, by virtue of an algorithm, harnesses the ability to learn from a given set of data points without having to explicitly program the rules of the domain, it falls under the ambit of machine learning.  

![AI-ML-DL](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png)

### Trinity of Machine Learning

<img style="float: center;" src="https://image.slidesharecdn.com/nextgentalk022015-150211154330-conversion-gate02/95/an-introduction-to-supervised-machine-learning-and-pattern-classification-the-big-picture-8-638.jpg?cb=1423785060">

To get a brief intro to Machine Learning & some basic concepts, you can refer to [this article](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471) on Medium by Adam Geitgey.

## Deep Learning
![DL_ML](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/ML_DL.png?raw=true)

Deep learning is a sub domain of Machine Learning that involves algorithms 'inspired' by the human brain, to tackle machine learning problems. So what is a Neural Network really? Imagine you're standing in front of a staircase. Every time you take a step up, you become a different person (some transformation in yourself takes place). By the time you get to the top, you've become a completely different person. Now think of these steps as layers, and every time we transition from one layer to another, a certain transformation in the input takes place. By the same analogy, a 'deep' neural network is the one with quite a number of these stairs. We'll cover the details on how a deep neural network trains in the upcoming articles.

### Everything Goes Deeper

Now, of course there isn't a single type of neural network because if it were so, well, then what's the hype about.

![NN_Type](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/neuralnetworks.png?raw=true)

Again, they key here is to not be overwhelmed, take it one step at a time, just like a Neural Network! Even though there are these numerous kinds of DNNs, much of the DL success can be attributed to some certain kinds of network architectures and their variants. For instance, most of the work done in the last decade or so in Computer Vision revolves around Convolutional Neural Networks, which we'll be covering in the next couple of articles among other things.

## How does FloydHub help?

Deep learning and its practise faces some practical difficulties. Most notably,
1. *Data is Primary, Learning is Secondary:* Datasets used to practise applications of DL like Computer Vision & NLP are huge. Take [this 'little' dataset](https://www.kaggle.com/c/yelp-restaurant-photo-classification/data) from Kaggle. That's nearly 6.54 Gigabytes of training data. But all you wanted, was to train a simple Neural Network in Python. So how do you deal with that?
2. *AI wants compute, AI wants to multitask:* Noticed how the AI bigwigs always talk about GPUs & Tensors? Well GPUs are great at running parallel tasks and training DL models requires a lot of that compute. So what's the problem? Well the state of the art GPUs required for the purpose don't come in handy and the entire system built on top of them costs very good money. Take a look at [Andrej Karpathy's gig](https://twitter.com/karpathy/status/648256662554341377), one of the pioneers of DL.

FloydHub takes care of those two things for you! It manages the data & gives you lots of compute with a minimalistic interface. To setup your own first project on FloydHub, refer to the [QuickStart Doc](https://docs.floydhub.com/getstarted/quick_start/) or head over to [this tutorial](https://blog.floydhub.com/getting-started-with-deep-learning-on-floydhub/).

## Summary

This was a high level introduction to the exciting field of Deep Learning (AI/ML/DL). Our hope is that this little introduction has inspired you to explore this domain of Deep Learning. In the upcoming series of articles, we'll take you through a journey from introducing PyTorch, a great DL framework built in Python to implementing CNNs & training them on the cloud, with large enough datasets and state of the art compute. What do you need? Well, a working internet connection & a zeal to explore. We'll take care of the rest.

Next up in this series: [Introduction to PyTorch](https://github.com/ReDeiPirati/intro-to-pytorch/blob/sw/beginner/1.PyTorch/PyTorch_intro.md)
