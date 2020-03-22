from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

dropout_value = 0.01
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block = 32
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 26
        # RF = 3

        # CONVOLUTION BLOCK 1

        # Input Block = 26
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 24
        # RF = 5

        # TRANSITION BLOCK 1

        # Input Block = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=35, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
        ) 
        # output_size = 22
        #RF = 7

        #Input = 22
        self.pool1 = nn.MaxPool2d(2, 2) 
        # output_size = 11
        #RF = 8


        # CONVOLUTION BLOCK 2

        #Input = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=45, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 9
        # RF = 12

        #Input = 9
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=55, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 7
        #RF = 16
        self.pool2 = nn.MaxPool2d(2, 2)
        #Input = 7
        
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=46, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 5
        # RF = 20

        # Input = 5
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) 
        # output_size = 3
        # RF = 24

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        self.fc = nn.Linear(512, 10) 
        self.fc1 = nn.Linear(64*9*9, 1024)

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x2 = self.convblock1(x)
        x3 = self.convblock2(torch.cat((x, x2), dim=1))
        #x_t1=torch.cat((x,x2, x3), dim=1)
        x4 =self.pool1(torch.cat((x,x2, x3), dim=1))
        x5=self.convblock3(x4)
        x6=self.convblock4(torch.cat((x4, x5), dim=1))
        x7=self.convblock5(torch.cat((x4, x5,x6), dim=1))
        x8=self.pool2(torch.cat((x5,x6, x7), dim=1))
        x9=self.convblock8(x8)
        x10=self.convblock6(torch.cat((x8, x9), dim=1))
        x11=self.convblock7(torch.cat((x8, x9,x10), dim=1))
        x12 = self.gap(x11)
        #x13 = self.convblock9(x12)
        #print(x13.shape)
        x13 = x12.view(-1, 10)
        #print(x13)
        return F.log_softmax(x13, dim=-1)
        #x13 =
        #x4 = self.convblock3(x)
        #x = self.pool1(x)
        #x = self.convblock4(x)
        #x = self.convblock5(x)
        #x = self.convblock6(x)
        #x = self.convblock7(x)
        #x = self.gap(x)        
        #x = self.convblock8(x)

        #x = x.view(-1, 10)
        #return F.log_softmax(x, dim=-1)