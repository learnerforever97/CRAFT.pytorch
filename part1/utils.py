import numpy as np
import cv2
import copy

def gauss_normal_generate(d):
    # generate normal gauss map
    width = d
    height = d
    center_x = width/2
    center_y = height/2
    # sigma principle to make the number at the edge of the circle to be very small
    sigma = d/3
    Gauss_map = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            dis = (i - center_y) ** 2 + (j - center_x) ** 2
            if dis > (d**2)/4:
                value = 0
            else:
                value = np.exp(-0.5 * dis / sigma ** 2)
            Gauss_map[i, j] = value
    return Gauss_map

def cvt2HeatmapImg(img):
    # display
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def cvt2HeatmapMatrix(img):
    # calculate
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def point_generate(x1, y1, x2, y2):
    x = []
    y = []
    mid_x1 = int((x1[0]+x1[2])/2)
    mid_y1 = int((y1[0]+y1[2])/2)
    mid_x2 = int((x2[0]+x2[2])/2)
    mid_y2 = int((y2[0]+y2[2])/2)
    x.append(copy.deepcopy(int((x1[0]+x1[1]+mid_x1)/3)))
    x.append(copy.deepcopy(int((x2[0]+x2[1]+mid_x2)/3)))
    x.append(copy.deepcopy(int((x2[2]+x2[3]+mid_x2)/3)))
    x.append(copy.deepcopy(int((x1[2]+x1[3]+mid_x1)/3)))
    y.append(copy.deepcopy(int((y1[0]+y1[1]+mid_y1)/3)))
    y.append(copy.deepcopy(int((y2[0]+y2[1]+mid_y2)/3)))
    y.append(copy.deepcopy(int((y2[2]+y2[3]+mid_y2)/3)))
    y.append(copy.deepcopy(int((y2[2]+y2[3]+mid_y2)/3)))
    return x, y

def interval_list_generate(text):
    word_list = []
    for part in text:
        part_word_list = part.strip().replace(' ', '\n').split('\n')
        for i in range(len(part_word_list)-1, -1, -1):
            if part_word_list[i] == '':
                part_word_list.remove('')
        word_list += part_word_list
    interval_i = 0
    interval_list = []
    for word in word_list:
        interval_i += len(word)
        interval_list.append(copy.deepcopy(interval_i))
    interval_list = interval_list[:-1]
    return interval_list

class averager(object):
    def __init__(self):
        self.reset()
    def add(self, v):
        count = v.numel()
        v = v.sum()
        self.n_count += count
        self.sum += v
    def reset(self):
        self.n_count = 0
        self.sum = 0
    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res