# FloydHub Introduction to Deep Learning: Transfer Learning

<p align="center">
<img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/trans_int.png?raw=true"/>
</p>

## Introduction

Hey, there! Welcome to part 4 in this mini series. As usual, before we begin, we'll go through some quick pointers about what we've done so far.

  1. We introduced PyTorch, a great new deep learning API for Python, along with many of its features including Tensors, optimization, loss functions etc.
  2. Introduced a classic deep learning problem: The Handwritten Digit Recognition task by MNIST. We then built two baseline models for the same -- One based on simple Logistic Regression & the other one based on a multi-layer perceptron. This is where we first introduced you to the structure of a basic Neural Network.
  3. We took a sneak peak at a ConvNet, neural network architecture specifically designed for translational invariance. We also gave a brief description about ImageNet Large Scale Visual Recognition Challenge & how Computer Vision wizards have built fantastic network architectures using ConvNets with state-of-the-art results on ImageNet.

In this article, following through our workflow, we introduce a new concept called **transfer learning**, where we leverage state-of-the-art models to make our own NanoConvNet better. In our last article, we talked briefly about these large and byzantine ConvNet architectures built by Google (Inception), Microsoft (ResNet) etc for the annual ILSVRC on the  ImageNet dataset. But these models are built over the course of weeks on sophisticated machines with lots of GPU compute. So our task here begins with a question -- can you, sitting on your tiny laptop & a much smaller dataset, ever get close to these high end models?

## From where should we start?

So you just acquired a new dataset, say [DogsVsCats](https://www.kaggle.com/c/dogs-vs-cats) from Kaggle (we've already got it all setup on Floyd, all you have to do is import & get going), with the following directory structure.

{% highlight python %}
.
.
├── train
	├── cats
	├── dogs
└── val
	├── cats
	├── dogs
└── test
.
.
{% endhighlight %}

access the same & perform some necessary transformations through native PyTorch modules -- essentially, prepare it for training.

```python
data_dir = '/dogsvscats'
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
```

<p align = "center">
<img src="https://blog.floydhub.com/static/catsdogs-0335375ee96d99c706971b50b926f9f2-f5f00.png"/>

*Note: Although we decided to go with a simple classification problem for illustration, Transfer Learning has much wider use cases as you'll see in some upcoming posts.*

**Now**, you have 2 options --

    1. Train your new ConvNet from scratch in PyTorch like we did last time to classify images of Dogs & Cats, experiment around with that architecture, lots of hyperparameter tuning & see where you get.
    2. **Leverage the work that's already been done***. In practise, very few people train a ConvNet from scratch (random initialisation), primarily because it is rare to get a good enough dataset and even more rare to find the compute to process all that information. So, using **pre-trained network weights** as initialisations or a fixed feature extractor helps in solving most of the problems in hand.

 Which one is the right way to go? [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/), Director of AI at Tesla & a Computer Vision wizard quoted the following piece of advise,

<p align="center">
<img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/karpathy_advise.png?raw=true"/>
 </p>

### Why does it work?

Much like in the case of ConvNets, these large complicated pre-trained networks try to detect shapes, contours, and some other high level image features. The only difference here is that they've been trained on a large enough dataset. Consider this example, if ResNet was trained on ImageNet which had over 50 categories of dogs and it was able to correctly identify what image corresponds to which category of dog, then it would be fairly simple for ResNet to work on a much simpler problem like identifying a Dog or a Cat.

<p align="center">
<img src = "http://ruder.io/content/images/2017/03/transfer_learning_setup.png"/>
</p>

### Approach

There are two ways to go about structuring a transfer-learning implementation after we've decided on a baseline i.e. from where we wish to learn from,

  1. **Fine-tuning**: Instead of random initializaion, we initialize the network with a pre-trained weights, like the one that is trained on ImageNet 1000 dataset. Rest of the training looks as usual.
  2. **Feature Extraction**: Here, we freeze the weights for all of the network's layers except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained (in terms of computed gradients).

For the purpose of our example, we've use ResNet as the pre-trained architecture. So let's get to the meat of Transfer Learning & implement it on our example in PyTorch. Interestingly, PyTorch supports all major pre-trained architectures. We encourage you to experiment with them during the exercise.

#### Fine-tuning

```python
model = models.resnet18(pretrained=True)
# Set class output
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion.cuda()
# ALL-Params get optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model, train_loss, train_acc = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
```

**Results**


#### Feature Extractor

```python
model = models.resnet18(pretrained=True)

# Fine Tuning - train only the last FC layer
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
if use_gpu:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
if use_gpu:
    criterion.cuda()
# ALL_Params being optimized
optimizer_ft = optim.SGD(model.fc.parameters(), lr=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model, train_loss, train_acc = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

**Results**

<p align="center">
<img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/tl_loss.png?raw=true"/>
<img src="https://github.com/sominwadhwa/sominwadhwa.github.io/blob/master/assets/intro_to_pytorch_series/tl_acc.png?raw=true"/>
</p>

As you could probably guess, since the model doesn't begin with randomly initialized weights, it converges much more quickly as compared to a vanilla ConvNet.

We strongly recommend & encourage you to try these examples out in our `ipython notebook` & implement the exercise on your own, may be fiddle with hyper-params or experiment with different architectures, in order to fully understand the depth associated with Transfer Learning.

### Fine-Tuning or Feature Extraction?

The choice really depends on the kind of data being used for the experiment. A few tips from our end are -

1. **Dataset used in your experiment is small in size & similar to the one on which you pre-trained model was trained on**-  
