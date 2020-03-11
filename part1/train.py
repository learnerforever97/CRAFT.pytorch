import torch
import scipy.io as sio
import numpy as np
import cv2
import copy
import argparse
import os
import time
import torch.optim as optim
from dataloader.dataset import ImageLoader_synthtext, collate
from utils import averager
from craft import CRAFT

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='SynthText')
parser.add_argument('--img_rootdir', default='/home/seg/dataset/SynthText/', type=str)
parser.add_argument('--gt_mat', default='/home/seg/dataset/gt.mat', type=str)
parser.add_argument('--go_on', default='', type=str)
parser.add_argument('--pre_model', default='', type=str)
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--store_sample', default='store', help='Where to store samples')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for critic')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=5000, help='Interval to be displayed')
args = parser.parse_args()

if not os.path.isdir(args.store_sample):
    os.system('mkdir {0}'.format(args.store_sample))

dataset = ImageLoader_synthtext(args)
assert dataset
data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, num_workers=4, shuffle=True, collate_fn=collate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.MSELoss(reduction='mean')
criterion = criterion.to(device)
craft = CRAFT(pretrained=True)

if args.go_on != '':
    print('loading pretrained model from %s' % args.pre_model)
    craft.load_state_dict(torch.load(args.pre_model), strict=False)
craft = craft.to(device)

loss_avg = averager()
optimizer = optim.Adam(craft.parameters(), lr=args.lr)

def train_batch(data):
    div = 10
    craft.train()
    img, char_label, interval_label = data
    img = img.to(device)
    char_label = char_label.to(device)
    interval_label = interval_label.to(device)

    img.requires_grad_()
    optimizer.zero_grad()
    preds, _ = craft(img)
    cost_char = criterion(preds[:,:,:,0], char_label).sum()/div
    cost_interval = criterion(preds[:,:,:,1], interval_label).sum()/div
    cost = cost_char + cost_interval
    cost.backward()
    optimizer.step()
    return cost

def main():
    for epoch in range(args.epoch):
        train_iter = iter(data_loader)
        i = 0
        while i < len(data_loader):
            time0 = time.time()
            data = train_iter.next()
            cost = train_batch(data)
            loss_avg.add(cost)
            i += 1

            # do checkpointing
            if i % args.saveInterval == 0:
                torch.save(craft.state_dict(), '{0}/craft_{1}_{2}_{3}.pth'.format(args.store_sample, epoch, i, loss_avg.val()))

            if i % args.displayInterval == 0:
                print('[%d/%d][%d/%d] lr: %.4f Loss: %f Time: %f s' %
                    (epoch, args.epoch, i, len(data_loader), optimizer.param_groups[0]['lr'], loss_avg.val(), time.time()-time0))
                loss_avg.reset()

if __name__ == '__main__':
	main()