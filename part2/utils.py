import copy
import cv2
import numpy as np
import torch
import math
import os

def thresh_calculate(img):
    h = img.shape[0]
    if h <= 40: 
        thresh = 190
    if 40 < h < 60:
        thresh = 190
    else:
        thresh = 190
    return thresh

def k_b_calculate(x_list, y_list):
    k_list = []
    b_list = []
    for i in range(4):
        if i == 3:
            i = -1
        if x_list[i+1]-x_list[i] != 0:
            k = (y_list[i+1]-y_list[i])/(x_list[i+1]-x_list[i])
            b = (x_list[i+1]*y_list[i]-y_list[i+1]*x_list[i])/(x_list[i+1]-x_list[i])    
        else:
            k = x_list[i]
            b = 99999  
        k_list.append(copy.deepcopy(k))
        b_list.append(copy.deepcopy(b))
    return k_list, b_list

def pixel_test(x, y, k_list, b_list):
    c_list = []
    for i in range(4):
        if b_list[i] == 99999:
            c = k_list[i]-x
        else:
            if i == 0 or i == 2:
                c = k_list[i]*x+b_list[i]-y
            else:
                c = (y-b_list[i])/k_list[i]-x
        c_list.append(copy.deepcopy(c))
    if c_list[0]<=0 and c_list[1]>=0 and c_list[2]>=0 and c_list[3]<=0:
        return True
    else:
        return False

def label_convert(label):
    label = label.cpu().detach().numpy()
    label = np.clip(label, 0, 255).astype(np.uint8)
    return label

def watershed_algorithm(img, thresh):
    _, img_t = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(img_t, kernel)
    _, contours, _ = cv2.findContours(img_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return(len(contours))

def label_generate(label_output, information):
    gauss_map = gauss_normal_generate(512)
    char_label = label_convert(label_output[:,:,0].squeeze())
    interval_label = label_convert(label_output[:,:,1].squeeze())
    char_label = cv2.resize(char_label, (1280, 720))
    interval_label = cv2.resize(interval_label, (1280, 720))
    char_label_all = np.zeros((720, 1280))
    interval_label_all = np.zeros((720, 1280))
    for i in range(len(information['tags'])):
        if information['tags'][i] == False:
            text_length = len(information['texts'][i])
            cor = information['polys'][i].astype(np.float32)
            x_list, y_list = cor[:,0], cor[:,1]
            x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
            w, h = x_max - x_min, y_max - y_min
            point = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
            m1 = cv2.getPerspectiveTransform(cor, point)
            char_label_line = cv2.warpPerspective(char_label, m1, (w, h))
            interval_label_line = cv2.warpPerspective(interval_label, m1, (w, h))
            thresh = thresh_calculate(char_label_line)
            peak_number = watershed_algorithm(char_label_line, thresh)
            if text_length == 0:
                alpha = -1
            else:
                alpha = (text_length - min(text_length, abs(text_length - peak_number)))/text_length
            if 0 <= alpha <= 0.6:
                char_label_line_new = np.zeros(char_label_line.shape)
                w_char = int(w/text_length)
                h_char = int(9*h/10)
                point_gauss = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype='float32')
                point_char = np.array([[0, 0], [w_char-1, 0], [w_char-1, h_char-1], [0, h_char-1]], dtype='float32')
                m = cv2.getPerspectiveTransform(point_gauss, point_char)
                char_gauss = cv2.warpPerspective(gauss_map, m, (w_char, h_char))
                for i in range(text_length):
                    for j in range(0, h_char):
                        for k in range(i*w_char, (i+1)*w_char):
                            if char_gauss[j][k-i*w_char] > char_label_line_new[j+int(h_char/18)][k]:
                                char_label_line_new[j+int(h_char/18)][k] = char_gauss[j][k-i*w_char] 
                char_label_line_new = cvt2HeatmapMatrix1(char_label_line_new)
                line_box = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
                target_box = np.array([[x_list[0], y_list[0]], [x_list[1], y_list[1]], [x_list[2], y_list[2]], [x_list[3], y_list[3]]], dtype='float32')
                M = cv2.getPerspectiveTransform(line_box, target_box)
                warped = cv2.warpPerspective(char_label_line_new, M, (1280, 720))
                char_label_all = np.where(warped > char_label_all, warped, char_label_all)

                interval_label_line_new = np.zeros(char_label_line.shape)
                w_char = int(w/text_length)
                h_char = int(3*h/4)
                point_gauss = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype='float32')
                point_char = np.array([[0, 0], [w_char-1, 0], [w_char-1, h_char-1], [0, h_char-1]], dtype='float32')
                m = cv2.getPerspectiveTransform(point_gauss, point_char)
                char_gauss = cv2.warpPerspective(gauss_map, m, (w_char, h_char))
                for i in range(text_length-1):
                    for j in range(0, h_char):
                        for k in range(i*w_char, (i+1)*w_char):
                            if char_gauss[j][k-i*w_char] > interval_label_line_new[j+int(h_char/6)][k+int(w_char/2)]:
                                interval_label_line_new[j+int(h_char/6)][k+int(w_char/2)] = char_gauss[j][k-i*w_char] 
                interval_label_line_new = cvt2HeatmapMatrix1(interval_label_line_new)
                line_box = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
                target_box = np.array([[x_list[0], y_list[0]], [x_list[1], y_list[1]], [x_list[2], y_list[2]], [x_list[3], y_list[3]]], dtype='float32')
                M = cv2.getPerspectiveTransform(line_box, target_box)
                warped = cv2.warpPerspective(interval_label_line_new, M, (1280, 720))
                interval_label_all = np.where(warped > interval_label_all, warped, interval_label_all)

            else:
                line_box = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
                target_box = np.array([[x_list[0], y_list[0]], [x_list[1], y_list[1]], [x_list[2], y_list[2]], [x_list[3], y_list[3]]], dtype='float32')
                M = cv2.getPerspectiveTransform(line_box, target_box)
                warped = cv2.warpPerspective(char_label_line, M, (1280, 720))
                char_label_all = np.where(warped > char_label_all, warped, char_label_all)

                line_box = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype='float32')
                target_box = np.array([[x_list[0], y_list[0]], [x_list[1], y_list[1]], [x_list[2], y_list[2]], [x_list[3], y_list[3]]], dtype='float32')
                M = cv2.getPerspectiveTransform(line_box, target_box)
                warped = cv2.warpPerspective(interval_label_line, M, (1280, 720))
                interval_label_all = np.where(warped > interval_label_all, warped, interval_label_all)

    char_label_all = cvt2HeatmapMatrix2(char_label_all)     
    interval_label_all = cvt2HeatmapMatrix2(interval_label_all)
    char_label_all = cv2.resize(char_label_all, (1120, 630))
    interval_label_all = cv2.resize(interval_label_all, (1120, 630))    
    return char_label_all, interval_label_all

def label_collate(label_output, information):
    char_labels, interval_labels, confidence_maps = [], [], []
    for index in range(label_output.shape[0]):    
        char_label, interval_label = label_generate(label_output[index], information[index]) 
        char_labels.append(copy.deepcopy(torch.FloatTensor(char_label)))
        interval_labels.append(copy.deepcopy(torch.FloatTensor(interval_label)))
    char_labels_stack = torch.stack(char_labels, 0)
    interval_labels_stack = torch.stack(interval_labels, 0)
    return char_labels_stack, interval_labels_stack

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

def getRecBoxes(textmap, linkmap, text_threshold, link_threshold, low_text):
    label_h, label_w = textmap.shape
    _, text_score = cv2.threshold(textmap, low_text, 255, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 255, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 255)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det = []
    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue
            
        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==255, text_score==0)] = 0   # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= label_w: ex = label_w
        if ey >= label_h: ey = label_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)
        det.append(box)
    return text_score_comb, det
    
def draw(img, det):
    ratio = 1280/1120
    for box in det:
        for i in range(4):
            if i == 3:
                i = -1
            cv2.line(img, (int(box[i][0]*ratio),int(box[i][1]*ratio)), (int(box[i+1][0]*ratio), int(box[i+1][1]*ratio)), (0,255,0), 2)
    return img

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

def cvt2HeatmapMatrix1(img):
    # calculate
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img

def cvt2HeatmapMatrix2(img):
    # calculate
    img = (np.clip(img, 0, 255)).astype(np.uint8)
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