from PIL import Image

def load_image(filename,size=None,scale=None):
    img = Image.open(filename)
    if size is not None:
        '''
                    参数值 	含义
            Image.NEAREST	低质量
            Image.BILINEAR	双线性
            Image.BICUBIC 	三次样条插值
            Image.ANTIALIAS	高质量
        '''
        img = img.resize((size,size),Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale),int(img.size[1]/scale)),Image.ANTIALIAS)

    return img

def save_image(filename,data):
    '''
    :param filename:
    :param data:  torch.Tensor
    :return:
    '''
    img = data.clone().clamp(0,255).numpy()
    img = img.transpose(1,2,0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

if __name__ == '__main__':
    img = load_image(filename='/home/wxx/pytorch_learning/fast_neural_style/image/content-image/amber.jpg',scale=2)
    img.save('new_img.png')