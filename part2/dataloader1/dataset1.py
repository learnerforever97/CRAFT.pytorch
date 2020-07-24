import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import torch.utils.data as data
from .synthtext import SynthText
import os
import sys
import random
sys.path.append("..")
from utils import interval_list_generate
from craft import CRAFT

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='SynthText')
parser.add_argument('--img_rootdir', default='/home/lbh/dataset/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/home/lbh/dataset/gt.mat', type=str)
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
args = parser.parse_args()

class ImageLoader1(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.dataset = SynthText(args)
    def __len__(self):
        return self.dataset.len()
    def __getitem__(self, index):
        img_path = args.img_rootdir + self.dataset.name[index][0]
        img, img_size = self.dataset.im_read_resize(img_path)
        char_label = self.dataset.char_label_generate(self.dataset.gauss_map_char, img_size, self.dataset.cor_list[index])
        interval_list = interval_list_generate(self.dataset.text[index])
        interval_label = self.dataset.interval_label_generate(self.dataset.gauss_map_interval, img_size, self.dataset.cor_list[index], interval_list)
        img, char_label, interval_label = random_augmentation(img, char_label, interval_label)
        char_label = cv2.resize(char_label, (384, 384))
        interval_label = cv2.resize(interval_label, (384, 384))
        img = torch.Tensor(img)
        char_label = torch.Tensor(char_label)
        interval_label = torch.Tensor(interval_label)
        return img, char_label, interval_label

def collate(batch):
    imgs = []
    char_labels = []
    interval_labels = []
    for sample in batch:
        imgs.append(sample[0])
        char_labels.append(sample[1])
        interval_labels.append(sample[2])
    imgs_stack = torch.stack(imgs, 0)
    char_labels_stack = torch.stack(char_labels, 0)
    interval_labels_stack = torch.stack(interval_labels, 0)
    return imgs_stack.permute(0,3,1,2), char_labels_stack, interval_labels_stack

def random_augmentation(image, char_label, interval_label):
    f = ImageTransfer(image, char_label, interval_label)
    seed = random.randint(0, 7)  # 0: original image used
    if 0 < seed < 7:
        methods = ['rotate', 'add_noise', 'change_contrast', 'change_hsv', 'center_crop', 'horizontal_flip']
        image, char_label, interval_label = getattr(f, methods[seed-1])()
    return image, char_label, interval_label

class ImageTransfer(object):
    """add noise, rotate, change contrast, change_hsv"""
    def __init__(self, image, char_label, interval_label):
        """image: a ndarray with size [h, w, 3]"""
        """label: a ndarray with size [h, w]"""
        self.image = image
        self.char_label = char_label
        self.interval_label = interval_label

    def add_noise(self):
        img = self.image * (np.random.rand(*self.image.shape) * 0.4 + 0.6)
        img = img.astype(np.uint8)
        char_label = self.char_label
        interval_label = self.interval_label
        return img, char_label, interval_label

    def rotate(self, angle=None, center=None, scale=1.0, angle_min=20, angle_max=90):
        h, w = self.image.shape[:2]
        if angle is None:
            angle = random.randint(angle_min, angle_max) if random.random() < 0.5 else random.randint(-angle_max, -angle_min)
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(self.image, M, (w, h)), cv2.warpAffine(self.char_label, M, (w, h)), cv2.warpAffine(self.interval_label, M, (w, h))

    def change_contrast(self):
        if random.random() < 0.5:
            k = random.randint(5, 9) / 10.0
        else:
            k = random.randint(11, 15) / 10.0
        b = 128 * (k - 1)
        img = self.image.astype(np.float)
        img = k * img - b
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        img = img.astype(np.uint8)
        char_label = self.char_label
        interval_label = self.interval_label
        return img, char_label, interval_label

    def change_hsv(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        char_label = self.char_label
        interval_label = self.interval_label
        s = random.random()
        def ch_h():
            dh = random.randint(2, 10) * random.randrange(-1,2,2)
            img[:, :, 0] = (img[:,:,0] + dh) % 180
        def ch_s():
            ds = random.random() * 0.25 + 0.7
            img[:, :, 1] = ds * img[:, :, 1]
        def ch_v():
            dv = random.random() * 0.35 + 0.6
            img[:, :, 2] = dv * img[:, :, 2]
        if s < 0.25:
            ch_h()
        elif s < 0.50:
            ch_s()
        elif s < 0.75:
            ch_v()
        else:
            ch_h()
            ch_s()
            ch_v()
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR), char_label, interval_label

    def center_crop(self):
        h, w = self.image.shape[0:-1]
        l = int(random.random() * 100)
        ymin = l
        ymax = h-l
        xmin = l
        xmax = w-l
        img = self.image[ymin:ymax, xmin:xmax, :]
        char_label = self.char_label[ymin:ymax, xmin:xmax]
        interval_label = self.interval_label[ymin:ymax, xmin:xmax]
        img = cv2.resize(img, (768, 768))
        return img, char_label, interval_label

    def horizontal_flip(self):
        img = np.flip(self.image, axis=1).copy()
        char_label = np.flip(self.char_label, axis=1).copy()
        interval_label = np.flip(self.interval_label, axis=1).copy()
        return img, char_label, interval_label

# def main():
#     dataset = ImageLoader(args)
#     data_loader = data.DataLoader(dataset, args.batch_size, num_workers=1, shuffle=True, collate_fn=collate)
#     model = CRAFT(pretrained=True).cuda()
#     for i, batch_samples in enumerate(data_loader):
#         batch_img, batch_char_label, batch_interval_label = batch_samples
#         batch_img, _ = model(batch_img.cuda())
#         print(i, batch_img.shape, batch_char_label.shape, batch_interval_label.shape)

# if __name__ == '__main__':
# 	main()