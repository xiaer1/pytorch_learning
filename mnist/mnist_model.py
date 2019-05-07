from __future__ import print_function,division
import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist_net(nn.Module):
    def __init__(self):
        super(mnist_net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=20,
                               kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.fc1 = nn.Linear(in_features=50*4*4,out_features=500)
        self.fc2 = nn.Linear(500,10)

    def forward(self, x):
        '''
        input x: (B,C,H,W) --> (B,1,28,28)
        conv1 = (28 - 5) / 1 + 1 = 24
        pool1 = 24 / 2 = 12
        conv2 = (12 - 5 ) /1 + 1 = 8
        pool2 = 8 / 2 = 4
        fc1 = 4 * 4 * inchannel
        '''
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1,4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)

