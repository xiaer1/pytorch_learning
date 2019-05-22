from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from six.moves import urllib
import tarfile
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class DatasetFromFolder(Dataset):

    def __init__(self,img_dir,input_transform=None,target_transform = None):
        super(DatasetFromFolder,self).__init__()

        self.img_filename = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_x = load_img(self.img_filename[index])
        target = input_x.copy()
        if self.input_transform:
            input_x = self.input_transform(input_x)
        if self.target_transform:
            target = self.target_transform(target)

        return input_x,target

    def __len__(self):
        return len(self.img_filename)
def load_img(file_path):
    img = Image.open(file_path).convert('YCbCr')
    y,cb,cr = img.split()
    # y.save('tmp/' + os.path.basename(file_path))
    return y
def is_image_file(file_name):
    # res=[True,False,False,False]  any(res) = True
    return any(file_name.endswith(ends) for ends in ['.png','.jpeg','.jpg','.PNG'])

def download_bsd300(dest='dataset'):
    output_img_dir = os.path.join(dest,"BSDS300/images")
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("download url",url)

        data = urllib.request.urlopen(url)

        file_path = os.path.join(dest,os.path.basename(url))

        with open(file_path,'wb') as f:
            f.write(data.read())

        print('Extracting data')
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item,dest)
        os.remove(file_path)
    return output_img_dir
def calculate_valid_crop_size(crop_size,upscale_factor):
    return crop_size - crop_size % upscale_factor

def input_transform(crop_size,upscale_factor):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor),
        transforms.ToTensor()
    ])

def target_transform(crop_size):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

def get_training_set(upscale_factor):
    output_img_dir = download_bsd300()
    train_dir = os.path.join(output_img_dir,'train')
    crop_size = calculate_valid_crop_size(256,upscale_factor)
    return DatasetFromFolder(train_dir,input_transform=input_transform(crop_size,upscale_factor),
                             target_transform=target_transform(crop_size))
def get_test_set(upscale_factor):
    output_img_dir = download_bsd300()
    test_dir = os.path.join(output_img_dir,'test')
    crop_size = calculate_valid_crop_size(256,upscale_factor)
    return DatasetFromFolder(test_dir, input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


if __name__ == '__main__':
    train_loader = get_training_set(upscale_factor=3)
    for x,y in train_loader:
        print(x.shape,y.shape)