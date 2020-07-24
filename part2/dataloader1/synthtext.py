import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import sys
import random
sys.path.append("..")
from utils import gauss_normal_generate_char, gauss_normal_generate_interval, cvt2HeatmapImg
from utils import cvt2HeatmapMatrix1, cvt2HeatmapMatrix2, point_generate, interval_list_generate

parser = argparse.ArgumentParser(description = 'SynthText')
parser.add_argument('--img_rootdir', default='/home/lbh/dataset/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/home/lbh/dataset/gt.mat',type=str)
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
        self.name = names[0]
        self.gauss_map_char = gauss_normal_generate_char(512, 3.34)
        self.gauss_map_interval = gauss_normal_generate_interval(512)
        self.gauss_size = 512

    def len(self):
        return len(self.data['charBB'][0])

    def im_read_resize(self, path):
        img = cv2.imread(path)
        img_size = (img.shape[0], img.shape[1])
        resized_img = cv2.resize(img, (768, 768))
        return resized_img, img_size

    def char_label_generate(self, gauss_map, img_size, cor_list):
        _, binary = cv2.threshold(gauss_map, 0.37 * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        # print(x, y, w, h)
        # (153, 153, 207, 207)
        regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        char_label = np.zeros(img_size)
        char_number = cor_list.shape[2]
        for i in range(char_number):
            x = []
            y = []
            for index in range(4):
                x.append(copy.deepcopy(cor_list[0][index][i]))
                y.append(copy.deepcopy(cor_list[1][index][i]))
            target_box = np.array([[x[0], y[0]], [x[1], y[1]],
                                   [x[2], y[2]], [x[3], y[3]]], dtype='float32')
            m = cv2.getPerspectiveTransform(regionbox, target_box)
            oribox = np.array([[[0, 0], [self.gauss_size - 1, 0], [self.gauss_size - 1, self.gauss_size - 1], [0, self.gauss_size - 1]]], dtype=np.float32)
            real_target_box = cv2.perspectiveTransform(oribox, m)[0]
            real_target_box = np.int32(real_target_box)
            real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, char_label.shape[1])
            real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, char_label.shape[0])
            if np.any(target_box[0] < real_target_box[0]) or (
                    target_box[3, 0] < real_target_box[3, 0] or target_box[3, 1] > real_target_box[3, 1]) or (
                    target_box[1, 0] > real_target_box[1, 0] or target_box[1, 1] < real_target_box[1, 1]) or (
                    target_box[2, 0] > real_target_box[2, 0] or target_box[2, 1] > real_target_box[2, 1]):
                warped = cv2.warpPerspective(gauss_map, m, (char_label.shape[1], char_label.shape[0]))
                warped = np.array(warped, np.uint8)
                char_label = np.where(warped > char_label, warped, char_label)
            else:
                xmin = real_target_box[:, 0].min()
                xmax = real_target_box[:, 0].max()
                ymin = real_target_box[:, 1].min()
                ymax = real_target_box[:, 1].max()
                width = xmax - xmin
                height = ymax - ymin
                _target_box = target_box.copy()
                _target_box[:, 0] -= xmin
                _target_box[:, 1] -= ymin
                _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
                warped = cv2.warpPerspective(gauss_map, _M, (width, height))
                warped = np.array(warped, np.uint8)
                if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                    print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                        ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                char_label[ymin:ymax, xmin:xmax] = np.where(warped > char_label[ymin:ymax, xmin:xmax], warped, char_label[ymin:ymax, xmin:xmax])
        char_label = cv2.resize(char_label, (768, 768))
        char_label = cvt2HeatmapMatrix1(char_label)
        return char_label

    def interval_label_generate(self, gauss_map, img_size, cor_list, interval_list):
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
            point1 = np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype='float32')
            point2 = np.array([[x[0]-x_min, y[0]-y_min], [x[1]-x_min, y[1]-y_min],
                            [x[2]-x_min, y[2]-y_min], [x[3]-x_min, y[3]-y_min]], dtype='float32')
            # point2 = top_left(point2)
            w_final = x_max - x_min
            h_final = y_max - y_min
            m = cv2.getPerspectiveTransform(point1, point2)
            target = cv2.warpPerspective(gauss_map, m, (w_final, h_final))
            for j in range(y_min, y_max):
                for k in range(x_min, x_max):
                    if target[j-y_min][k-x_min] > interval_label[j][k]:
                        interval_label[j, k] = target[j-y_min][k-x_min]       
        interval_label = cv2.resize(interval_label, (768, 768))
        interval_label = cvt2HeatmapMatrix2(interval_label)
        return interval_label

def random_augmentation(image, char_label, interval_label):
    f = ImageTransfer(image, char_label, interval_label)
    # seed = random.randint(0, 7)  # 0: original image used
    seed = 6
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
#     synthtext = SynthText(args)
#     print(synthtext.len())
#     i = 600
#     img_path = args.img_rootdir + synthtext.name[i][0] 
#     print(img_path)
#     print(synthtext.text[i])
#     img, img_size = synthtext.im_read_resize(img_path)
#     # img has been resized
#     char_label = synthtext.char_label_generate(synthtext.gauss_map_char, img_size, synthtext.cor_list[i])
#     # print(synthtext.cor_list[i].transpose(2, 1, 0))
#     interval_list = interval_list_generate(synthtext.text[i])
#     interval_label = synthtext.interval_label_generate(synthtext.gauss_map_interval, img_size, synthtext.cor_list[i], interval_list)
#     img, char_label, interval_label = random_augmentation(img, char_label, interval_label)
#     char_label = cv2.resize(char_label, (768, 768))
#     interval_label = cv2.resize(interval_label, (768, 768))
#     char_label = cv2.applyColorMap(char_label, cv2.COLORMAP_JET)
#     interval_label = cv2.applyColorMap(interval_label, cv2.COLORMAP_JET)
#     cv2.imwrite('pic/p1.jpg', img)
#     cv2.imwrite('pic/p2.jpg', char_label)
#     cv2.imwrite('pic/p3.jpg', interval_label)

# if __name__ == '__main__':
# 	main()