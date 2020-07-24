import torch
import scipy.io as sio
import os
import numpy as np
from math import *
import cv2
import copy
import random
import argparse
import torch.utils.data as data
from .ic15 import ICDAR2015
# import sys
# sys.path.append("..")
# from craft import CRAFT

class ImageLoader2(data.Dataset):
    def __init__(self, config):
        self.dataset = ICDAR2015(config['image_path'], config['is_training'])
    def __len__(self):
        return self.dataset.len()
    def __getitem__(self, index):
        img_path, information = self.dataset.image_path_list[index], self.dataset.targets[index]
        img = self.dataset.im_read_resize1(img_path)
        img_path_new = img_path.replace('train_images', 'train_images_d')
        img_new = self.dataset.im_read_resize2(img_path_new)
        img = torch.FloatTensor(img)
        img_new = torch.FloatTensor(img_new)
        return img, img_new, information

def collate2(batch):
    imgs = []
    img_news = []
    informations = []
    for sample in batch:
        imgs.append(copy.deepcopy(sample[0]))
        img_news.append(copy.deepcopy(sample[1]))
        informations.append(copy.deepcopy(sample[2]))
        imgs_stack = torch.stack(imgs, 0)
        img_news_stack = torch.stack(img_news, 0)
    return imgs_stack.permute(0, 3, 1, 2), img_news_stack.permute(0, 3, 1, 2), informations

def batch_random_augmentation(img, char_label, interval_label):
    img = img.permute(0, 2, 3, 1).detach().numpy()
    char_label = char_label.detach().numpy()
    interval_label = interval_label.detach().numpy()

    f = ImageTransfer(img, char_label, interval_label)
    seed = random.randint(0, 7)  # 0: original image used
    if 0 < seed < 7:
        methods = ['rotate', 'rotate', 'add_noise', 'change_contrast', 'change_hsv', 'center_crop', 'horizontal_flip']
        img, char_label, interval_label = getattr(f, methods[seed-1])()
    return img, char_label, interval_label

class ImageTransfer(object):
    """add noise, rotate, change contrast, change_hsv"""
    def __init__(self, img, char_label, interval_label):
        """img: a ndarray with size [b, h, w, 3]"""
        """label: a ndarray with size [b, h/2, w/2]"""
        self.img = img
        self.char_label = char_label
        self.interval_label = interval_label

    def add_noise(self):
        img = np.zeros(self.img.shape)
        char_label = self.char_label
        interval_label = self.interval_label
        for i in range(self.img.shape[0]): 
            img[i] = self.img[i] * (np.random.rand(*self.img[i].shape) * 0.4 + 0.6)
        return img, char_label, interval_label

    def rotate(self, angle=None, center=None, scale=1.0, angle_min=20, angle_max=80):
        b = self.img.shape[0]
        h, w = self.img.shape[1:3]
        h1, w1 = self.char_label.shape[1:]
        if angle is None:
            k = random.random()
            if k < 0.5:
                angle = random.randint(angle_min, angle_max)  
            else:    
                angle = random.randint(-angle_max, -angle_min)
        if center is None:
            center = (w // 2, h // 2)
            center1 = (w1 // 2, h1 // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M1 = cv2.getRotationMatrix2D(center1, angle, scale)
        img = np.zeros(self.img.shape)
        char_label = np.zeros(self.char_label.shape)
        interval_label = np.zeros(self.interval_label.shape)
        for i in range(b): 
            img[i] = cv2.warpAffine(self.img[i], M, (w, h))
            char_label[i] = cv2.warpAffine(self.char_label[i], M1, (w1, h1))
            interval_label[i] = cv2.warpAffine(self.interval_label[i], M1, (w1, h1))
        return img, char_label, interval_label

    def change_contrast(self):
        img = np.zeros(self.img.shape)
        char_label = self.char_label
        interval_label = self.interval_label
        if random.random() < 0.5:
            k = random.randint(5, 9) / 10.0
        else:
            k = random.randint(11, 15) / 10.0
        b = 128 * (k - 1)
        for i in range(self.img.shape[0]): 
            img[i] = self.change_contrast_add(self.img[i], k, b)
        return img, char_label, interval_label

    def change_contrast_add(self, img, k, b):
        img = img.astype(np.float)
        img = k * img - b
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)
        return img

    def change_hsv(self):
        img = np.zeros(self.img.shape)
        char_label = self.char_label
        interval_label = self.interval_label
        for i in range(self.img.shape[0]): 
            img[i] = self.change_hsv_add(self.img[i])
        return img, char_label, interval_label

    def change_hsv_add(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def center_crop(self):
        img = np.zeros(self.img.shape)
        char_label = np.zeros(self.char_label.shape)
        interval_label = np.zeros(self.interval_label.shape)
        l = int(random.random() * 500)

        for i in range(self.img.shape[0]): 
            img[i] = self.center_crop_add1(self.img[i], l)
            char_label_resize = cv2.resize(self.char_label[i], (2240, 1260))
            interval_label_resize = cv2.resize(self.interval_label[i], (2240, 1260))
            char_label[i] = self.center_crop_add2(char_label_resize, l)
            interval_label[i] = self.center_crop_add2(interval_label_resize, l)
        return img, char_label, interval_label

    def center_crop_add1(self, img, l):
        h, w = img.shape[0:-1]
        ymin = 0
        ymax = h
        xmin = l
        xmax = w-l
        img = img[ymin:ymax, xmin:xmax, :]
        img = cv2.resize(img, (2240, 1260))
        return img

    def center_crop_add2(self, img, l):
        h, w = img.shape
        ymin = 0
        ymax = h
        xmin = l
        xmax = w-l
        img = img[ymin:ymax, xmin:xmax]
        img = cv2.resize(img, (1120, 630))
        return img

    def horizontal_flip(self):
        img = np.flip(self.img, axis=2).copy()
        char_label = np.flip(self.char_label, axis=2).copy()
        interval_label = np.flip(self.interval_label, axis=2).copy()
        return img, char_label, interval_label
