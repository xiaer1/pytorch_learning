from __future__ import print_function
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

def parse_super_resolve():
    parse = argparse.ArgumentParser(description='pytorch Super Res Usage')
    parse.add_argument('--input_img',type=str,required=True,help='input image')
    parse.add_argument('--model',type=str,required=True,help='saved checkpoint')
    parse.add_argument('--output_img',type=str,help='stored path')
    parse.add_argument('--cuda',default=0,help='use cuda?')

    return parse.parse_args()

def main():
    opt = parse_super_resolve()
    img = Image.open(opt.input_img).convert('YCbCr')
    y,cb,cr = img.split()
    cb.save('cb.png')
    cr.save('cr.png')
    y.save('y.png')
    model = torch.load(f=opt.model)
    # print(model)
    transform = transforms.ToTensor()
    X = transform(y).view(1,-1,y.size[1],y.size[0])
    if opt.cuda:
        model = model.cuda()
        X = X.cuda()
    out = model(X)
    out = out[0].detach().numpy() * 255.0
    out = out.clip(0,255)
    img_y = Image.fromarray(np.uint8(out[0]),mode='L')
    img_y.save('img_y.png')
    img_cb = cb.resize(img_y.size,Image.BICUBIC)
    img_cr = cr.resize(img_y.size,Image.BICUBIC)
    out_img = Image.merge(mode='YCbCr',bands=[img_y,img_cb,img_cr]).convert('RGB')

    out_img.save(opt.output_img)
if __name__ == '__main__':
    main()