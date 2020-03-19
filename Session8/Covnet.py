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





def test():
    net = Net()
    y = net(torch.randn(1,3,32,32))
    print(y.size())  

  

