import copy
import cv2
import numpy as np
import torch
import math
import os
import torch
from craft import CRAFT
from utils import getRecBoxes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pretrained_model_path = "/home/lbh/part2/store/craft_ic15_16.pth"
model = CRAFT(pretrained=True).cuda()
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()

for index in range(1, 501):
    print(index)
    img_path = '/home/lbh/dataset/icdar2015/test_images/img_' + str(index) + '.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (2240, 1260))
    img = torch.FloatTensor(img).cuda().permute(2, 0, 1).unsqueeze(0)
    output, _ = model(img)
    char_label = output[:,:,:,0].squeeze()
    char_label = char_label.cpu().detach().numpy()
    char_label = np.clip(char_label, 0, 255).astype(np.uint8)
    interval_label = output[:,:,:,1].squeeze()
    interval_label = interval_label.cpu().detach().numpy()
    interval_label = np.clip(interval_label, 0, 255).astype(np.uint8)
    _, det = getRecBoxes(char_label, interval_label, 120, 40, 90)
    ratio = 1280/1120
    txt_path = '/home/lbh/part2/result/res_img_' + str(index) + '.txt'
    with open(txt_path, 'a', encoding='UTF-8') as f:
        for box in det:
            for i in range(4):
                if i == 3:
                    f.write(str(int(box[i][0]*ratio)))
                    f.write(',')
                    f.write(str(int(box[i][1]*ratio)))    
                else:
                    f.write(str(int(box[i][0]*ratio)))
                    f.write(',')
                    f.write(str(int(box[i][1]*ratio)))
                    f.write(',')
            f.write('\r\n')
        f.close()
