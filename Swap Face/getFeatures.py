import cv2
import skimage.feature as ski
import matplotlib.pyplot as plot
import numpy as np
import math
import pdb

def getFeatures(img, minX, maxX, minY, maxY):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_bbox = gray_img[minY:maxY, minX:maxX]
    corner_response = ski.corner_harris(img_bbox)
    peaks = ski.peak_local_max(corner_response, 24)

    #corner_response = cv2.cornerHarris(img_bbox,2,3,0.04)
    #indices = np.where(corner_response > np.mean(corner_response))
    feature_pts = np.concatenate((peaks[:, 1].reshape(peaks.shape[0], 1), peaks[:, 0].reshape(peaks.shape[0], 1)), 1)
    feature_pts[:, 0] += minX
    feature_pts[:, 1] += minY
    return feature_pts