import cv2
import skimage.feature as ski
import matplotlib.pyplot as plot
import numpy as np
import math

'''
  File name: getFeatures.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''


def getFeatures(img, bbox):
    # TODO: Your code here
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    num_bbox = np.shape(bbox)[0];
    peak_res = []
    max_num_features = 0;
    for i in range(num_bbox):
        minX = min([bbox[i][0][0], bbox[i][1][0], bbox[i][2][0], bbox[i][3][0]])
        maxX = max([bbox[i][0][0], bbox[i][1][0], bbox[i][2][0], bbox[i][3][0]])
        minY = min([bbox[i][0][1], bbox[i][1][1], bbox[i][2][1], bbox[i][3][1]])
        maxY = max([bbox[i][0][1], bbox[i][1][1], bbox[i][2][1], bbox[i][3][1]])

        img_bbox = gray_img[minY: maxY, minX: maxX]
        corner_response = ski.corner_harris(img_bbox)
        #minDistance1 = int((bbox[i][3][1] - bbox[i][0][1])/12.0)
        #minDistance2 = int((bbox[i][3][0] - bbox[i][0][0])/12.0)
        #minDistance = max(min(minDistance1, minDistance2), 1);
        peaks = ski.peak_local_max(corner_response, 5)
        #plot.imshow(img_bbox)
        #plot.scatter(peaks[:, 1], peaks[:, 0], c='r')
        #plot.show()
        # add offsets
        peaks[:, 0] = bbox[i][0][1] + peaks[:, 0]
        peaks[:, 1] = bbox[i][0][0] + peaks[:, 1]
        peak_res.append(peaks)
        if max_num_features < np.shape(peaks)[0]:
            max_num_features = np.shape(peaks)[0]

    x = np.zeros([max_num_features, num_bbox])
    y = np.zeros([max_num_features, num_bbox])

    for i in range(num_bbox):
        num_current_peak = np.shape(peak_res[i])[0]
        x[0:num_current_peak, i] = peak_res[i][:, 0]
        y[0:num_current_peak, i] = peak_res[i][:, 1]
    #plot.imshow(img)
    #plot.scatter(y, x, c='r')
    #plot.show()

    return y, x
