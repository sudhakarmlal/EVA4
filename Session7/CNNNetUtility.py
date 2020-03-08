from __future__ import print_function
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def getTransformer():
  transform = transforms.Compose(
  [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  return transform

def trainset():
  transform = transforms.Compose(
  [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
  return trainset

def testset():
  transform = transforms.Compose(
  [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
  return testset

def getclasses():
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return classes



# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images

#trainset = trainset()
#testset = testset()
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                        shuffle=False, num_workers=2)

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
#classes = getclasses()
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value)
        ) # output_size = 36 , RF= 5*5

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 18,  RF= 6*6

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, groups=128 , bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value)
        ) # output_size = 20 , RF= 14*14
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 10,  RF= 16*16
        
        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=3, dilation = 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=3, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_value)
        ) # output_size = 12,  RF= 48*48
        self.pool3 = nn.MaxPool2d(2, 2) # output_size = 6,  RF= 56*56
                     
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool2(x)
        x = self.convblock3(x)
        x = self.pool3(x)
        #print(x.shape)               
        x = self.gap(x)       
        x = self.convblock5(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

def getDevice():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device



def getOptimizer():
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  return optimizer



def testImages(testloader):
  dataiter = iter(testloader)
  images, labels = dataiter.next()

# print images
  imshow(torchvision.utils.make_grid(images))
  print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
  return images

