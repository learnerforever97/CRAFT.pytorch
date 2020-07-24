import os
import numpy as np
import cv2
import sys
import io
import copy
import torch
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.path.append("..")
# from craft import CRAFT

def load_ann(gt_paths):
    res = []
    for gt in gt_paths:
        item = {}
        item['polys'] = []
        item['tags'] = []
        item['texts'] = []
        item['paths'] = gt
        reader = open(gt, encoding='utf-8').readlines()
        for line in reader:
            parts = line.strip().strip('\xef\xbb\xbf').split(',')
            label = parts[-1]
            line = [i.strip('\ufeff') for i in parts]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            item['polys'].append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            item['texts'].append(label)
            if label == '###':
                item['tags'].append(True)
            else:
                item['tags'].append(False)
        item['polys'] = np.array(item['polys'], dtype=np.float32)
        item['tags'] = np.array(item['tags'], dtype=np.bool)
        item['texts'] = np.array(item['texts'], dtype=np.str)
        res.append(item)
    return res

class ICDAR2015(object):
    def __init__(self, path, is_training = True):
        self.is_training = is_training
        self.generate_information(path)
    def generate_information(self, path):
    	if self.is_training:
            image_folder = os.path.join(path, 'train_images')
            gt_folder = os.path.join(path, 'train_gts')
            self.image_path_list = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]
            gt_path_list = [os.path.join(gt_folder, gt) for gt in os.listdir(gt_folder)]
            self.image_path_list = sorted(self.image_path_list)
            gt_path_list = sorted(gt_path_list)
            self.targets = load_ann(gt_path_list)
    	else:
    		image_folder = os.path.join(path, 'test_images')
    		self.image_path_list = [os.path.join(image_folder, image) for image in os.listdir(image_folder)]
    		self.image_path_list = sorted(self.image_path_list)
    def len(self):
    	return len(self.image_path_list)
    def im_read_resize1(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (2240, 1260))
        return img
    def im_read_resize2(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (1280, 720))
        return img