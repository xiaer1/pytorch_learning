import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()

        vgg_pretrained_features = models.vgg16(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(name=str(x),module=vgg_pretrained_features[x])

        for x in range(4,9):
            self.slice2.add_module(name=str(x),module=vgg_pretrained_features[x])

        for x in range(9,16):
            self.slice3.add_module(name=str(x),module=vgg_pretrained_features[x])
        for x in range(16,23):
            self.slice4.add_module(name=str(x),module=vgg_pretrained_features[x])

        for param in self.parameters():
            '''
            Parameters 是 torch.Tensor 的subclass, 特别之处在于:当和 Module 连用时(被assign
            Module 属性,会自动添加到 parameter list 中,封装成 Module.parameters 迭代器.
            属性:
                data(Tensor)
                requires_grad(bool): Default: True
            '''
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h  =self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple("VggOutputs",
                    ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2,h_relu2_2,h_relu3_3,h_relu4_3)

        return out

if __name__ == '__main__':
    vgg = Vgg16()
    x = torch.rand(size=(100,3,28,28))
    vgg(x)