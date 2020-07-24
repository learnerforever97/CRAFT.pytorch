import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import os
import time
import torch.optim as optim
from dataloader2.dataset2 import ImageLoader2, collate2, batch_random_augmentation
from craft import CRAFT
from utils import thresh_calculate, k_b_calculate, pixel_test, label_convert 
from utils import watershed_algorithm, label_generate, label_collate, averager
from dataloader1.dataset1 import ImageLoader1, collate1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='ICDAR2015')
parser.add_argument('--img_rootdir', default='/home/lbh/dataset/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/home/lbh/dataset/gt.mat', type=str)
parser.add_argument('--go_on', default='go_on', type=str)
parser.add_argument('--pre_model1', default="/home/lbh/part1/store/craft_synthtext_best.pth", type=str)
parser.add_argument('--pre_model2', default="/home/lbh/part2/store/craft_ic15_16.pth", type=str)
parser.add_argument('--batch_size1', type=int, default=4, help='input batch size')
parser.add_argument('--batch_size2', type=int, default=1, help='input batch size')
parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for critic')
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--displayInterval1', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--displayInterval2', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
args = parser.parse_args()

os.system('mkdir {0}'.format(args.store_sample))

dataset1 = ImageLoader1(args)
assert dataset1
data_loader1 = torch.utils.data.DataLoader(dataset1, args.batch_size1, num_workers=4, shuffle=None, collate_fn=collate1)

config={'is_training':True, 'image_path':'/home/lbh/dataset/icdar2015'}
dataset2 = ImageLoader2(config)
assert dataset2
data_loader2 = torch.utils.data.DataLoader(dataset2, args.batch_size2, num_workers=4, shuffle=True, collate_fn=collate2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.MSELoss(reduction='mean')
criterion = criterion.to(device)
craft1 = CRAFT(pretrained=True)
craft2 = CRAFT(pretrained=True)
if args.go_on != '':
    print('loading pretrained model from %s' % args.pre_model1)
    print('loading pretrained model from %s' % args.pre_model2)
    craft1.load_state_dict(torch.load(args.pre_model1), strict=False)
    craft2.load_state_dict(torch.load(args.pre_model2), strict=False)
craft1 = craft1.to(device)
craft2 = craft2.to(device)
loss_avg = averager()
optimizer = optim.Adam(craft2.parameters(), lr=args.lr)

def train_batch1(data):
    craft2.train()
    img, char_label, interval_label = data
    img = img.to(device)
    char_label = char_label.to(device)
    interval_label = interval_label.to(device)

    img.requires_grad_()
    optimizer.zero_grad()
    preds, _ = craft2(img)
    cost_char = criterion(preds[:,:,:,0], char_label).sum()
    cost_interval = criterion(preds[:,:,:,1], interval_label).sum()
    cost = cost_char + cost_interval
    cost.backward()
    optimizer.step()
    return cost

def train_batch2(data):
    craft1.eval()
    craft2.train()
    img, img_new, information = data
    
    img_new = img_new.to(device)
    label_output, _ = craft1(img_new)
    char_label, interval_label = label_collate(label_output, information)
    
    img, char_label, interval_label = batch_random_augmentation(img, char_label, interval_label)
    img = torch.FloatTensor(img).permute(0, 3, 1, 2).to(device)
    char_label = torch.FloatTensor(char_label).to(device)
    interval_label  = torch.FloatTensor(interval_label).to(device)

    img.requires_grad_()
    optimizer.zero_grad()
    preds, _ = craft2(img)
    cost_char = criterion(preds[:,:,:,0], char_label).sum()
    cost_interval = criterion(preds[:,:,:,1], interval_label).sum()
    cost = cost_char + cost_interval
    cost.backward()
    optimizer.step()
    return cost

def main():
    for epoch in range(args.epoch):
        if epoch % 30 == 0:
            train_iter1 = iter(data_loader1)
            i = 0
            while i < int(1/10*len(data_loader1)):
                time0 = time.time()
                data = train_iter1.next()
                cost = train_batch1(data)
                loss_avg.add(cost)
                i += 1

                if i % args.displayInterval1 == 0:
                    print('[%d/%d][%d/%d] lr: %.4f Loss: %f Time: %f s' %
                        (epoch, args.epoch, i, int(1/10*len(data_loader1)), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
                    loss_avg.reset()

        else:
            train_iter2 = iter(data_loader2)
            i = 0
            while i < len(data_loader2):
                time0 = time.time()
                data = train_iter2.next()
                cost = train_batch2(data)
                loss_avg.add(cost)
                i += 1

                # do checkpointing
                if i % args.saveInterval == 0 and epoch % 30 == 29:
                    torch.save(craft2.state_dict(), '{0}/craft_{1}_{2}_{3}.pth'.format(args.store_sample, epoch, i, loss_avg.val()))

                if i % args.displayInterval2 == 0:
                    print('[%d/%d][%d/%d] lr: %.6f Loss: %f Time: %f s' %
                        (epoch, args.epoch, i, len(data_loader2), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
                    loss_avg.reset()
    
if __name__ == '__main__':
	main()