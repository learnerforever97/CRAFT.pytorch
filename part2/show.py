import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import os
from craft import CRAFT

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pretrained_model_path = "/home/lbh/part2/store/craft_ic15_16.pth"
    img_path = "/home/lbh/dataset/icdar2015/test_images/img_160.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (2240, 1260))
    cv2.imwrite('picture/ori_img.jpg', img)
    model = CRAFT(pretrained=True).cuda()
    model.load_state_dict(torch.load(pretrained_model_path))
    model.eval()

    img = torch.FloatTensor(img).cuda().permute(2, 0, 1).unsqueeze(0)
    output, _ = model(img)
    char_label = output[:,:,:,0].squeeze()
    char_label = char_label.cpu().detach().numpy()
    char_label = np.clip(char_label, 0, 255).astype(np.uint8)
    char_label = cv2.applyColorMap(char_label, cv2.COLORMAP_JET)
    interval_label = output[:,:,:,1].squeeze()
    interval_label = interval_label.cpu().detach().numpy()
    interval_label = np.clip(interval_label, 0, 255).astype(np.uint8)
    interval_label = cv2.applyColorMap(interval_label, cv2.COLORMAP_JET)
    cv2.imwrite('picture/char.jpg', char_label)
    cv2.imwrite('picture/interval.jpg', interval_label)

if __name__ == '__main__':
	main()