# -*- coding: utf-8 -*-

## https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from IPython import display
DATA_FOLDER = './torch_data/VGAN/MNIST'

from utils import Logger

def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

def images_to_vectors(images):
    return images.view(images.size(0), 784)
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda() 
    return n
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 1024),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.hidden1 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.hidden2 = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
                )
        self.out = nn.Sequential(
                torch.nn.Linear(256, n_out),
                torch.nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.LeakyReLU(0.2)
                )
        self.hidden1 = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2)
                )
        self.hidden2 = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.2)
                )
        self.out = nn.Sequential(
                torch.nn.Linear(1024, n_out),
                torch.nn.Tanh()
                )
        
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
        
        

discriminator = DiscriminatorNet()
generator = GeneratorNet()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()

# optimizer
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# loss function
loss = nn.BCELoss()
# epochs
num_epochs = 200
d_steps = 1

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # reset grad
    optimizer.zero_grad()
    
    # 1. train on real data
    prediction_real = discriminator(real_data)
    # cal error and backprop.
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()
    
    # 2. train on fake data
    prediction_fake = discriminator(fake_data)
    
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 3. update weights
    optimizer.step()
    
    # return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    
    # reset grad
    optimizer.zero_grad()
    
    # sample noise and generate data
    prediction = discriminator(fake_data)
    
    # calculate error
    error = loss(prediction, ones_target(N))
    error.backward()
    
    # update
    optimizer.step()
    
    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 200

## Start training

for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(data_loader):
        
        # 1. train D
        real_data = Variable(images_to_vectors(real_batch))
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # gen fake data
        fake_data = generator(noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        
        # 2. Train G
        # gen fake data again
        fake_data = generator(noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)
        
        # Display Progress
        if (n_batch) % 100 == 0:
            display.clear_output(True)
            # Display Images
            test_images = vectors_to_images(generator(test_noise)).data.cpu()
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
        # Model Checkpoints
        logger.save_models(generator, discriminator, epoch)
        
    