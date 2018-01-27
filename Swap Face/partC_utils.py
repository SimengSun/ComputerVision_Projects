import numpy as np
import matplotlib.pyplot as plot
import copy
import PyNet as net
import pdb
import cv2
import skimage.transform as skitrans

def preprocess(img):
    '''
    :param img: h x w x 3
    :return: ori_img
             images: 1 x 3 x 40 x 40
    '''
    images = []
    ori_imgs = []
    ori_img = copy.deepcopy(img)
    ori_imgs.append(ori_img)
    img = img.transpose(2, 0, 1)
    img -= np.array([[[119.78]], [[99.50]], [[89.93]]])
    img /= 255.0
    images.append(img)

    images = np.asarray(images)
    ori_imgs = np.asarray(ori_imgs)

    h = images.shape[2]
    w = images.shape[3]

    images = net.upsample2d(images, (40, 40))

    return ori_imgs, images

def getLandMarks(img, x, y, h, w):
    '''

    :param img: 1 x 5 x 40 x 40
    :return: marks: 5 x 2
    '''

    marks = []
    for i in range(img.shape[1]):
        img[0][i][0:3, :] = 0
        img[0][i][:, 37:40] = 0
        img[0][i][37:40, :] = 0
        img[0][i][:, 0:3] = 0

        maxid = np.unravel_index(np.argmax(img[0][i]), (40, 40))
        maxid = np.array(maxid)
        print maxid
        # swap
        temp = maxid[0]
        maxid[0] = maxid[1]
        maxid[1] = temp

        maxid[0] = maxid[0]*w/40.0+x
        maxid[1] = maxid[1]*h/40.0+y
        marks.append(maxid)
    marks = np.asarray(marks)
    return marks

def drawMarks(img, marks):
    img_res = copy.deepcopy(img)
    for i in range(marks.shape[0]):
        cv2.circle(img_res, (int(round(marks[i][0])), int(round(marks[i][1]))), 3, (255, 0, 0), 4)
    return img_res

def drawBbox(img, minX, maxX, minY, maxY):
    img_res = copy.deepcopy(img)
    cv2.line(img_res, (minX, minY), (minX, maxY), (0, 255, 0), 3)
    cv2.line(img_res, (minX, minY), (maxX, minY), (0, 255, 0), 3)
    cv2.line(img_res, (maxX, maxY), (minX, maxY), (0, 255, 0), 3)
    cv2.line(img_res, (maxX, maxY), (maxX, minY), (0, 255, 0), 3)

    return img_res

def getOffsetPts(mark_pts, minX, minY):
    offset_pts = copy.deepcopy(mark_pts)
    offset_pts[:, 0] -= minX
    offset_pts[:, 1] -= minY
    return offset_pts
