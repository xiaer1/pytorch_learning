import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet,self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3,32,kernel_size = 9,stride = 1)
        self.in1 = nn.InstanceNorm2d(num_features=32,affine=True)

        self.conv2 = ConvLayer(32,64,kernel_size=3,stride=2)
        self.in2 = nn.InstanceNorm2d(num_features=64,affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(num_features=128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = ConvLayer(128,64,kernel_size=3,stride=1,upsample=2)
        self.in4 = nn.InstanceNorm2d(num_features=64,affine=True)
        self.deconv2 = ConvLayer(64,32,kernel_size=3,stride=1,upsample=2)
        self.in5 = nn.InstanceNorm2d(32,affine=True)
        self.deconv3 = ConvLayer(32,3,kernel_size=9,stride=1)

        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu( self.in1(self.conv1(x)) )
        x = self.relu( self.in2(self.conv2(x)) )
        x = self.relu(self.in3(self.conv3(x)))

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.relu(self.in4(self.deconv1(x)))
        x = self.relu(self.in5(self.deconv2(x)))
        x = self.deconv3(x)
    
        return x

class ResidualBlock(nn.Module):
    def __init__(self,channel):
        super(ResidualBlock,self).__init__()
        self.conv1 = ConvLayer(channel,channel,kernel_size=3,stride=1)
        self.in1 = nn.InstanceNorm2d(num_features=channel,affine=True)

        self.conv2 = ConvLayer(channel, channel, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(num_features=channel, affine=True)

        self.relu = nn.ReLU()
    def forward(self, x):

        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual

        return out


class ConvLayer(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,upsample=None):
        super(ConvLayer,self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        #left, right, top, bottom
        #self.padding = nn.ReflectionPad2d((2,3,4,5)
        self.reflection_padding = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channel,out_channel,kernel_size,stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = F.interpolate(x_in,mode='nearest',scale_factor=self.upsample)
        out = self.reflection_padding(x_in)
        out = self.conv2d(out)

        return out

if __name__ == '__main__':
    x = torch.rand((100,3,28,28))
    model = TransformerNet()
    out = model(x)