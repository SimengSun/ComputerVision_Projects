'''
  File name: p3_dataloader.py
  Author:
  Date:
'''

import numpy as np
import PyNet as net
import scipy.ndimage as scimage
import pdb
import matplotlib.pyplot as plot
import skimage.color as skicol
import copy

def load_list():
    '''
    :return: img path list, lable list
    '''
    with open('./MTFL/training.txt', 'r') as f:
        lines = f.readlines()
    img_lst = []
    label_lst = []
    for line in lines:
        sp = line.split()
        if len(sp) < 1:
            continue
        img_lst.append(sp[0])
        label_lst.append(sp[1:11])
    return img_lst, label_lst

def load_test_list():
    with open('./MTFL/testing.txt', 'r') as f:
        lines = f.readlines()
    img_lst = []
    label_lst = []
    for line in lines:
        sp = line.split()
        if len(sp) < 1:
            continue
        img_lst.append(sp[0])
        label_lst.append(sp[1:11])
    return img_lst, label_lst


def preprocess(img_lst, label_lst):
    '''
    :param img_lst:
    :param label_lst:
    :return: img: 4d ndarray, label:4d
    '''

    images = []
    ori_imgs = []
    # read image
    for fname in img_lst:
        fname = fname.replace('\\', '/')
        img = scimage.imread('MTFL/'+fname).astype('float64')
        if(len(img.shape) < 3):
            img = skicol.gray2rgb(img)
        ori_img = copy.deepcopy(img)
        ori_imgs.append(ori_img)
        #pdb.set_trace()
        img = img.transpose(2,0,1)
        img -= np.array([[[119.78]], [[99.50]], [[89.93]]])
        img /= 255.0
        images.append(img)

    images = np.asarray(images)
    ori_imgs = np.asarray(ori_imgs)
    # get gt
    label_lst = np.asarray(label_lst).astype('float64')
    h = images.shape[2]
    w = images.shape[3]
    labels = net.get_gt_map(label_lst, h, w)
    images = net.upsample2d(images, (40, 40))
    labels = net.upsample2d(labels, (40, 40))

    return ori_imgs, images, labels

