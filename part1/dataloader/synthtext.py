import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import sys
sys.path.append("..")
from utils import gauss_normal_generate, cvt2HeatmapImg
from utils import cvt2HeatmapMatrix, point_generate, interval_list_generate

parser = argparse.ArgumentParser(description = 'SynthText')
parser.add_argument('--img_rootdir', default='/home/seg/dataset/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/home/seg/dataset/gt.mat',type=str)
args = parser.parse_args()

class SynthText(object):
    def __init__(self, args):
        self.args = args
        self.generate_information()
        
    def generate_information(self):
        self.data = sio.loadmat(self.args.gt_mat)
        char_BB = self.data['charBB']
        self.cor_list = char_BB[0]
        img_txt = self.data['txt']
        self.text = img_txt[0]
        names = self.data['imnames']
        # the third 0 to get the string in list
        self.name = names[0]
        self.gauss_map = gauss_normal_generate(20)

    def len(self):
        return len(self.data['charBB'][0])

    def im_read_resize(self, path):
        img = cv2.imread(path)
        img_size = (img.shape[0], img.shape[1])
        if img_size[0] > img_size[1]:
            img = np.rot90(img, -1)
        resized_img = cv2.resize(img, (600, 400), cv2.INTER_NEAREST)
        return resized_img, img_size

    def char_label_generate(self, gauss_map, img_size, cor_list):
        # generate the first map with all char box being replaced with gauss map
        h = img_size[0]
        w = img_size[1]
        char_label = np.zeros((h, w))
        char_number = cor_list.shape[2]
        for i in range(char_number):
            x = []
            y = []
            for index in range(4):
                x.append(copy.deepcopy(int(cor_list[0][index][i])))
                y.append(copy.deepcopy(int(cor_list[1][index][i])))
            x_min = max(min(x), 0)
            x_max = min(max(x), w)
            y_min = max(min(y), 0)
            y_max = min(max(y), h)
            point1 = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype='float32')
            point2 = np.array([[x[0]-x_min, y[0]-y_min], [x[1]-x_min, y[1]-y_min],
                            [x[2]-x_min, y[2]-y_min], [x[3]-x_min, y[3]-y_min]], dtype='float32')
            w_final = x_max - x_min
            h_final = y_max - y_min
            m = cv2.getPerspectiveTransform(point1, point2)
            target = cv2.warpPerspective(gauss_map, m, (w_final, h_final), cv2.INTER_NEAREST)
            for j in range(y_min, y_max):
                for k in range(x_min, x_max):
                    if target[j-y_min][k-x_min] > char_label[j][k]:
                        char_label[j, k] = target[j-y_min][k-x_min]
        if h > w:
            char_label = np.rot90(char_label, -1)
        char_label = cv2.resize(char_label, (300, 200), cv2.INTER_NEAREST)
        char_label = cvt2HeatmapMatrix(char_label)
        return char_label
    
    def interval_label_generate(self, gauss_map, img_size, cor_list, interval_list):
        # generate the first map with all char box being replaced with gauss map
        h = img_size[0]
        w = img_size[1]
        interval_label = np.zeros((h, w))
        char_number = cor_list.shape[2]
        for i in range(char_number-1):
            if i+1 in interval_list:
                continue
            x1 = []
            y1 = []
            x2 = []
            y2 = []
            for index in range(4):
                x1.append(copy.deepcopy(int(cor_list[0][index][i])))
                y1.append(copy.deepcopy(int(cor_list[1][index][i])))
                x2.append(copy.deepcopy(int(cor_list[0][index][i+1])))
                y2.append(copy.deepcopy(int(cor_list[1][index][i+1])))
            x, y = point_generate(x1, y1, x2, y2)
            x_min = max(min(x), 0)
            x_max = min(max(x), w)
            y_min = max(min(y), 0)
            y_max = min(max(y), h)
            point1 = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], dtype='float32')
            point2 = np.array([[x[0]-x_min, y[0]-y_min], [x[1]-x_min, y[1]-y_min],
                            [x[2]-x_min, y[2]-y_min], [x[3]-x_min, y[3]-y_min]], dtype='float32')
            w_final = x_max - x_min
            h_final = y_max - y_min
            m = cv2.getPerspectiveTransform(point1, point2)
            target = cv2.warpPerspective(gauss_map, m, (w_final, h_final), cv2.INTER_NEAREST)
            for j in range(y_min, y_max):
                for k in range(x_min, x_max):
                    if target[j-y_min][k-x_min] > interval_label[j][k]:
                        interval_label[j, k] = target[j-y_min][k-x_min]
        if h > w:
            interval_label = np.rot90(interval_label, -1)        
        interval_label = cv2.resize(interval_label, (300, 200), cv2.INTER_NEAREST)
        interval_label = cvt2HeatmapMatrix(interval_label)
        return interval_label

# def main():
#     synthtext = SynthText(args)
#     print(synthtext.len())
#     for i in range(synthtext.len()):
#         img_path = args.img_rootdir + synthtext.name[i][0] 
#         img, img_size = synthtext.im_read_resize(img_path)
#         # img has been resized
#         char_label = synthtext.char_label_generate(synthtext.gauss_map, img_size, synthtext.cor_list[i])
#         interval_list = interval_list_generate(synthtext.text[i])
#         interval_label = synthtext.interval_label_generate(synthtext.gauss_map, img_size, synthtext.cor_list[i], interval_list)
#         print(img.shape, char_label.shape, interval_label.shape)

# if __name__ == '__main__':
# 	main()