# FloydHub Introduction to Deep Learning

### Abstract

[(Prof) Andrew Ng, Coursera co-founder & Stanford Adjunct Professor, recently said that AI is the new electricity](https://youtu.be/21EiKfQYZXc), [even Google shifted its focus from mobile-first to AI-first](https://youtu.be/Y2VF8tmLFHw?t=6m50s), even the VCs with their startups are riding this wave of hype. Some even think of it as magic while others are really scared, primarily due to the premature sci-fi literature. But what is it really? With [Zuck vs Elon declarations](https://www.theguardian.com/technology/2017/jul/25/elon-musk-mark-zuckerberg-artificial-intelligence-facebook-tesla) on twitter and the narrative about human job replacement due to AI, we are often overwhelmed to an extent that we overlook the countless applications of AI that make ourselves so much more productive and motivates some of the brightest minds around the world to work in this area.

At FloydHub we firmly believe that, if widely understood and tested, this technology has the potential to change human-machine interaction providing additional value to the society. One of our fundamental goal is to `#makeAIsimple`. We want to build a platform providing dreamers & developers a toolbox to create, learn, and experiment & see *what change can they bring about.*. Shall we begin? ;)

Table of Contents:
	- [AI](#artificial-intelligence)
	- [ML](#machine-learning)
	- [DL](#deep-learning)
	- [Narrow AI](#narrow-ai)
	- [General AI](#general-ai)
	- [what's next?](#)
	- [Summary](#summary)

## Artificial Intelligence

Everyone likes to talk about the miraculous effectiveness of AI: AI can "see", AI can "understand" and AI can "whatsoever-a-people-can-do". Is this all true? Well if you are riding the the wave of hype created around the latest breakthroughs such as AlphaGo Zero or Google's self driving car, all of it may seem too abstruse, and at the same time make us overlook the humanly-fathomable yet elegant solutions built using AI by developers sitting their rooms all around the world. For instance,

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

Human species is one of the most incredibly things ever created in term of complexity and this is the fundamentally the reason behind the successes in this field. Solving the "intelligence mystery" means unboxing human complexity and finally know ourselves in our fullyness.

### Fields in AI

![AI_fields](https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/AI.png?raw=true)

AI is an extremely multi-disciplinary field, it covers: Planning, Robotics, Math, Neuroscience, Biology, Statistic, Natural Language Processing and Understanding, Knowledge Representation, Reasoning, and Machine Learning. Do not worry if you have not a solid understanding of these fields, during your journey, with us ;), you will learn a lot.

With the last achievement in this field, you can try to improve all the existing technologies and maybe simplify the human-machine interaction building a future where human possibilities will be so wide that the only limit is our immagination.

### Is AI an old field

Yes, even if we have not discussed the full history behind AI, it's good to know that the name AI was created by John McCarthy who spent all his life on it. But again we are not here to spend words on AI-story, we are here to understand what's behind this field and the recent breakthough. Before introducing the next building block we have to underline a thing.

### Knowledge Representation

![Knowledge representation image, rule engine inference]

The key component of reasoning, is to have someting to reasoning on! In the past, but even now, AI was/is reduced to hardcode structured informations in some computer representable format so that we can reason on it. With reason I mean: defining rules by which an algorithm can infer information from. Moreover we can explore the information represented using appropriate data structure such as graph or tree, and reduce the reasoning or planning task to a visit on this structures.

![translate a task to a visit on graph or tree]

It's easy to recognize that this is not the way we work in term of a possible knowledge pipeline. We acquire knowledge with experience: more the experiences we face more the knowledge we gain. In other terms: we learn knowledges, we have not any hardcoded knowledge. This bring us to the next building block.

## Machine Learning

Put simply, when a machine, by virtue of an algorithm, harnesses the ability to learn from a given set of data points without having to explicitly program the rules of the domain, it falls under the ambit of machine learning.  

![AI-ML-DL](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png)

### Trinity of Machine Learning

<img style="float: center;" src="https://image.slidesharecdn.com/nextgentalk022015-150211154330-conversion-gate02/95/an-introduction-to-supervised-machine-learning-and-pattern-classification-the-big-picture-8-638.jpg?cb=1423785060">

#### Supervised Learning
(student preparing for exam example: Train are assignments, Evaluation is exam)

#### Unsupervised Learning
the Yann LeCun cake: Twitter/Facebook post] [toddler in the world, he create useful representation of the world without any prior knownledge or label]

#### Reinforcement Learning
the Yann LeCun cake: Twitter/Facebook post] [toddler in the world, he learn interacting with the env.]

### ML workflow

#### Create a Dataset

#### Choose your Model

#### Train

#### Evaluate

### ML is an Optimization Problem


## Deep Learning

key concepts: NN rebranding, gpus(computational power) and representational learning

### What is a NN

### Everything goes deeper

Deep SL, Deep UL, DRL

### DL success

#### Mainstream reasons of adoption
![Andrew Ng NIPS tutorial 2016]

Transfer Learning
Representational Learning
More Data, More Accuracy



#### Real world applications

Self-driving car
precision medicine(rethinopathy and skin cancer detection)
Alpha Go
Conversational Agent
System reccomandation, Spam filter etc...


## Narrow AI

Current technology is capable to solve things without required a new way of interconnect knowledge like repetitive task and human task around 1-10s. Do one job at superhuman performance.


## General AI

No one knows how can we reach it.
The challenge:
 - Catastrophic forgetting (A new task delete the previous knowledge, how can we overcome this?)
 - Continuous Learning (in which way is organized the knowledge inside our brain?),
 - Safe AI (Are ML model, safe in term of cyber security(Adversarial example), but even not expected behavior? Example RL which exploit env.)
 - Unbiased Dataset (Machine and model are neither sexist nor racist, but their data can be, how can we unbiased ds?)

We need a Cognitive Toolkit to evaluate this model. (See Karpathy slide)

## What's next?

As just said above, the road to achieve GAI is not defined, even if we have not a well defined path, here a list of the most interesting topics which are pushing researcher to find a way to achieve GAI and may be a another step in the direction of the Intelligence Algorithm which drive our evolutionary behavior.

[Is Backpropagation the right algorithm to drive the learning process?](https://www.axios.com/ai-pioneer-advocates-starting-over-2485537027.html) This is a question of [Geoffry Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) who recently become suspicious about the famous algo behind our learning models. I do not know if backprop is the right way to drive learning, but it's really notable and admirable when one of the man who have driven the research in this discipline has the courage to questioning their own work and try with a new formulation.

![slide 26 LeCun NIPS 2016 Predictive Learning]

[Unsupervised Learning is the cake](https://www.facebook.com/yann.lecun/posts/10153426023477143). Humans and animals do not learn from labeled data, this is main reason because unsupervised learning is so hot on Academia: researcher think that Unsupervised Learning is the key to learn good representation of data on which supervised and reinforcement learning will compute.

[AI = RL + DL](http://icml.cc/2016/tutorials/deep_rl_tutorial.pdf). This is a quote from [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Home.html), one of the main architech behind AlphaGo, Deep Reinforcement Learning is another very active research field where top AI companies are challeging each other(DeepMind, OpenAI etc...). Raising the bar in this field means having an Agent which is able to move into an enviroment and reaching is goals.

![Le Cun citation about GANs](https://cdn-images-1.medium.com/max/2000/1*AZ5-3WdNdYyC2U0Aq7RhIg.png)

[Generative Adversarial Network]() is the most interesting idea in the last ten years in machine learning. We have already discussed the amount of data needed to train our DL models. GANs is a really cool class of models whose purpose is to generate high quality data similar to the dataset distribution which they are learning. Since collecting dataset is extremely consuming in term of time and money resource, simulating/generating data is the only feasible solution. By instance: [Waymo is massively using simulation to train its automous systems](https://www.theatlantic.com/technology/archive/2017/08/inside-waymos-secret-testing-and-simulation-facilities/537648/), [Apple is using it to refine syntethic images](https://machinelearning.apple.com/2017/07/07/GAN.html), and other are pointing in the same direction. In order to train very deep model, you need a lot of high quality data.

[Prior Consciusness](https://arxiv.org/abs/1709.08568). Y. Bengio proposal of consciusness with our current knowledge and technolgy. [Follow this link for a great explanation.](https://www.quora.com/What-is-Yoshua-Bengios-new-Consciousness-Prior-paper-about)

[Opening the black box of deep neural networks via information](https://arxiv.org/abs/1703.00810). Prof. Shwartz-Ziv and Naftali Tishby have tried to explain what's happening during training. This research underline the following things:
- SGD involves 2 distinct phase: Memorization and Compression. Memorization phase: High Mean, Low Variance (few epochs), Compression phase: Low Mean, High Variance(a lot of epochs). This can be translate as: during the first step, the Information Plane of each Layer is adjust in a similar way to memorize the task, then each leayer begin to exclude all the irrelevant information(compression phase)
- the number of hidden layer reduces the training time to reach optimal compression and generalization
Follow this [link for a seminar on this work by Naftali Tishby](https://www.youtube.com/watch?v=FSfN2K3tnJU)

* There is a **hidden** chapter just below, can you find it?* (Gamification)

(hidden html element)
### What about AI apocalypse
Once we reach GAI it could be possible that can cause our extinctions?
Report image about who is supporting and who is not supporting
Elon vs Zuck
Siraj :)

## Summary

This was a high level introduction to the exciting field of Deep Learning (AI/ML/DL).
Our hope is that this article have inspired you as much as it has pushed us to build FloydHub and allow AI-folks and You to take our present in the future you have dreamed of.
