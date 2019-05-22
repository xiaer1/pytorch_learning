import torch
import torch.nn as nn
import torch.nn.init as init

class SuperResolutionNet(nn.Module):
    def __init__(self,upscale_factor):
        super(SuperResolutionNet,self).__init__()

        self.relu = nn.ReLU()
        # (w - f + 2 * p) / s + 1
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(5,5),
                               stride=(1,1),padding=(2,2))
        self.conv2 = nn.Conv2d(64,64,(3,3),(1,1),(1,1))
        self.conv3 = nn.Conv2d(64,32,(3,3),(1,1),(1,1))
        self.conv4 = nn.Conv2d(32,upscale_factor**2,(3,3),(1,1),(1,1))
        '''
        Input: (N,C∗upscale_factor2,H,W)
        Output: (N,C,H∗upscale_factor,W∗upscale_factor)
        '''
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight,gain=init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, gain=init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, gain=init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)