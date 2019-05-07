from __future__ import print_function,division

from mnist_model import mnist_net
from data_loader import get_dataloader

from argparse import Namespace
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

args = Namespace(
    batch_size = 64,
    test_batch_size = 1000,
    epochs = 10,
    lr = 0.01,
    momentum = 0.5,
    use_cuda = False,
    seed = 1,
    log_interval = 10,
    save_model = True
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='train.log', level=logging.DEBUG, format=LOG_FORMAT)

def train(model,train_loader,optimizer,device,log_interval,epoch):
    model.train()

    for idx,(data,labels) in enumerate(train_loader):
        data,labels = data.to(device),labels.to(device)
        optimizer.zero_grad()

        output = model(data)
        #CrossEntropyLoss()=log_softmax() + NLLLoss()
        loss = F.nll_loss(output,labels)

        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%)]\t Loss:{:.6f}'.format(epoch,
                    idx * len(data),len(train_loader.dataset),100*idx / len(train_loader),
                                                          loss.item()))
            logging.info('Train Epoch: {} [{} / {} ({:.0f}%)]\t Loss:{:.6f}'.format(epoch,
                    idx * len(data),len(train_loader.dataset),100*idx / len(train_loader),
                                                          loss.item()))

def test(model,test_loader,device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,labels in test_loader:
            # data,labels = data.to(device),labels.to(device)
            output = model(data)

            test_loss += F.nll_loss(output,labels).item()

            pred = output.argmax(dim=1,keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n Test set: Average loss:{:.4f},Accuracy: {} / {} ({:.0f}%)'.format(
        test_loss,correct,len(test_loader.dataset),100 * correct / len(test_loader.dataset)
    ))

    logging.info('\n Test set: Average loss:{:.4f},Accuracy: {} / {} ({:.0f}%)'.format(
        test_loss,correct,len(test_loader.dataset),100 * correct / len(test_loader.dataset)
    ))
def main():
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader,test_loader = get_dataloader(args.batch_size,args.test_batch_size)

    model = mnist_net().to(device)
    optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)
    for epoch in range(1,args.epochs+1):
        train(model,train_loader,optimizer,device,args.log_interval,epoch)
        test(model,test_loader,device)

    if args.save_model:
        torch.save(model.state_dict(),'mnist_cnn.pt')
if __name__ == '__main__':
    main()
