from __future__ import print_function,division

import argparse
import numpy as np
from PIL import Image
import torch
import re
import utils
from torchvision import transforms
from style_model import TransformerNet
from vgg import Vgg16
from torchvision import models
from vggNet import Vgg16

#train on ImageNet,too long
def train(args):
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.mul(255))
    ])
    style = utils.load_image(args.style_image,args.style_size)
    style = style_transform(style)
    style = style.repeat(1,1,1,1)
    vgg = Vgg16()
    # print(style.shape)
    # mean = style.new_tensor([0.485, 0.456, 0.406]).view(-1,1,1)
    # style = style.div_(255)
    # print(mean.shape)
    # t = style - mean

    feature_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in feature_style]

    # for j,gram in enumerate(gram_style):
    #
    #     for i in range(1):
    #         img = gram[i].mul(2550).clone().clamp(0, 255).numpy().astype('uint8')
    #         print(img)
    #         # img = img.transpose(1, 2, 0).astype('uint8')
    #         img = Image.fromarray(img)
    #         img.save(str(j)+'.png')
    print(feature_style.relu2_2)
def stylize(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content,scale=args.content_scale)

    content_transform = transforms.Compose([
        transforms.ToTensor(),  #automatic convert to [0,1]
        #torch.Tensor -- > mul
        transforms.Lambda(lambda x:x.mul(255)),
        # transforms.ToPILImage(),
        # transforms.Resize(size=(500,500))
    ])
    content_image = content_transform(content_image) #numpy
    content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for keys in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$',keys):
                del state_dict[keys]
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        output = style_model(content_image)
        utils.save_image(args.output_image,output[0])

def main():
    parser = argparse.ArgumentParser(description="parser for fast-nerual-style")
    #====================================================
    #using exists model to style transfer
    #python neural_style.py --content image/content-image/amber.jpg --model saved_models/udnie.pth --cuda 0 --output_image style_udnie_scale2.png
    #====================================================
    # parser.add_argument('--content',type=str,required=True,
    #                     help='path to content image')
    # parser.add_argument('--content_scale',type=float,default=0.7,
    #                     help='factor for scaling down the content image')
    # parser.add_argument('--output_image',type=str,required=True,
    #                     help='path to save the out put image')
    # parser.add_argument('--model',type=str,required=True,
    #                     help='saved model to be used for stylizing the image')
    # parser.add_argument('--cuda',type=int,required=True,
    #                     help='set it to 1 running on GPU,0 for CPU')

    #============================================================
    #train own model
    #============================================================
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of training epochs, default is 2")
    parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    parser.add_argument("--dataset", type=str, required=False,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    parser.add_argument("--style-image", type=str, default="image/style-images/rain-princess.jpg",
                                  help="path to style-image")
    parser.add_argument("--save-model-dir", type=str, required=False,
                                  help="path to folder where trained model will be saved.")
    parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    args = parser.parse_args()

    # stylize(args)
    train(args)
if __name__ == '__main__':
    main()
