import numpy as np
import skimage.transform as skt
import math
import pdb

'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes
    
    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    # TODO: Your code here
    newbbox = bbox
    XsList = []
    YsList = []
    maxLength = -1
    for i in range(np.shape(bbox)[0]):
        similarityTrans = skt.SimilarityTransform()
        startxx, startyy, newxx, newyy = startXs[:, i], startYs[:, i], newXs[:, i], newYs[:, i]
        # cull by distance
        distance = ((newyy - startyy) * (newyy - startyy) + (newxx - startxx) * (newxx - startxx))
        threshold = 49
        startxx = startxx[distance < threshold]
        startyy = startyy[distance < threshold]
        newxx = newxx[distance < threshold]
        newyy = newyy[distance < threshold]
        # cull by borders
        startxx = startxx[startxx >= 0]
        startyy = startyy[startyy >= 0]
        newxx = newxx[newxx >= 0]
        newyy = newyy[newyy >= 0]

        startxx, startyy = np.array([startxx]).transpose(), np.array([startyy]).transpose()
        newxx, newyy = np.array([newxx]).transpose(), np.array([newyy]).transpose()

        src = np.concatenate((startxx, startyy), axis=1)
        dst = np.concatenate((newxx, newyy), axis=1)

        similarityTrans.estimate(src, dst)
        homography = similarityTrans.params

        #pdb.set_trace()
        bbox3 = np.concatenate((bbox[i], np.ones([4, 1])), axis=1)
        bbox3 = np.transpose(bbox3)

        newbbox3 = np.matmul(homography, bbox3)
        newbbox3 = np.transpose(newbbox3)
        newbbox[i] = np.round(newbbox3[0:4, 0:2])

        XsList.append(newxx)
        YsList.append(newyy)
        if newxx.shape[0] > maxLength :
            maxLength = newxx.shape[0]

    Xs = np.zeros([maxLength, np.shape(bbox)[0]])
    Ys = np.zeros([maxLength, np.shape(bbox)[0]])

    for i in range(np.shape(bbox)[0]):
        Xs[0:np.shape(XsList[i])[0], i] = XsList[i][0:np.shape(XsList[i])[0], 0]
        Ys[0:np.shape(YsList[i])[0], i] = YsList[i][0:np.shape(YsList[i])[0], 0]

    return Xs, Ys, newbbox