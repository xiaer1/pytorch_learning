from __future__ import print_function,division
from torchvision import datasets,transforms
import torch

def get_dataloader(batch_size,test_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,),std=(0.3081,))
        ])

    train_dataset = datasets.MNIST(root='./data',train=True,
                                   transform=transform,download=True)

    train_loader =  torch.utils.data.DataLoader(train_dataset,
                                    batch_size = batch_size,shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False,
                                   transform=transform, download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=test_batch_size, shuffle=True)
    
    return train_loader,test_loader

if __name__ == '__main__':
    train_loader,test_loader = get_dataloader(32,1000)
    for data,labels in test_loader:
        print(data.shape,labels.shape)

