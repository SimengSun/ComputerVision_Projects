import numpy as np
import cv2
import estimateFeatureTranslation
from scipy import signal
import matplotlib.pyplot as plot
import math
import pdb

'''
  File name: estimateAllTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for all features for each bounding box as well as its four corners
    - Input startXs: all x coordinates for features wrt the first frame
    - Input startYs: all y coordinates for features wrt the first frame
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newXs: all x coordinates for features wrt the second frame
    - Output newYs: all y coordinates for features wrt the second frame
'''

def estimateAllTranslation(startXs, startYs, img1, img2):
    # TODO: Your code here
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # xx, yy = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    # dd = np.sqrt(xx * xx + yy * yy)
    # gaussian_2d = np.exp(- ((dd - 0) ** 2 / (2.0 * (0.5 ** 2))))
    #
    # #gaussian_2d = GaussianPDF_2D(0, 0.05, 8, 8)
    # # x gradient
    # gx = np.array([[-1, 1], [-1, 1]])
    # # y gradient
    # gy = np.array([[-1, 1], [-1, 1]]).transpose()
    # # conv(x, 2d gaussian)
    # gx_gaussian = signal.convolve2d(gaussian_2d, gx, 'same')
    # # conv(y, 2d gaussian)
    # gy_gaussian = signal.convolve2d(gaussian_2d, gy, 'same')
    # pdb.set_trace()
    # # conv(I, gaussian_x)
    # I2x = signal.convolve2d(gray_img2, gx_gaussian, 'same')
    # # conv(I, gaussian_y)
    # I2y = signal.convolve2d(gray_img2, gy_gaussian, 'same')
    #
    # I1x = signal.convolve2d(gray_img1, gx_gaussian, 'same')
    # # conv(I, gaussian_y)
    # I1y = signal.convolve2d(gray_img1, gy_gaussian, 'same')
    #
    # I1M = np.sqrt(I1x*I1x + I1y*I1y)
    # I2M = np.sqrt(I2x*I2x + I2y*I2y)
    #
    # # conv(I, gaussian_x)
    # I2Mx = signal.convolve2d(I2M, gx_gaussian, 'same')
    # # conv(I, gaussian_y)
    # I2My = signal.convolve2d(I2M, gy_gaussian, 'same')
    #
    # I1Mx = signal.convolve2d(I1M, gx_gaussian, 'same')
    # # conv(I, gaussian_y)
    # I1My = signal.convolve2d(I1M, gy_gaussian, 'same')

    # cv2.imshow("I1M", I1M)
    # cv2.imshow("I2M", I2M)
    # cv2.imshow("I2x", I2x)
    # cv2.imshow("I2y", I2y)
    # cv2.imshow("I1x", I1x)
    # cv2.imshow("I1y", I1y)
    # cv2.imshow("I2Mx", I2Mx)
    # cv2.imshow("I2My", I2My)
    # cv2.imshow("I1Mx", I1Mx)
    # cv2.imshow("I1My", I1My)
    # cv2.waitKey(0)
    G = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])
    gaussian_matrix = (1.0 / 159.0) * G
    # x gradient
    gx = np.array([[1, -1], [1, -1]])
    # y gradient
    gy = np.array([[1, -1], [1, -1]]).transpose()
    # conv(x, 2d gaussian)
    gx_gaussian = signal.convolve2d(gaussian_matrix, gx, 'same')
    # conv(y, 2d gaussian)
    gy_gaussian = signal.convolve2d(gaussian_matrix, gy, 'same')
    #dx, dy = np.gradient(gaussian_matrix)
    #pdb.set_trace()
    I1x = signal.convolve2d(gray_img1, gx_gaussian, 'same');
    I1y = signal.convolve2d(gray_img1, gy_gaussian, 'same');

    I2x = signal.convolve2d(gray_img2, gx_gaussian, 'same');
    I2y = signal.convolve2d(gray_img2, gy_gaussian, 'same');

    I1M = np.sqrt(I1x * I1x + I1y * I1y)
    I2M = np.sqrt(I2x * I2x + I2y * I2y)

    # conv(I, gaussian_x)
    I2Mx = signal.convolve2d(I2M, gx_gaussian, 'same')
    # conv(I, gaussian_y)
    I2My = signal.convolve2d(I2M, gy_gaussian, 'same')

    I1Mx = signal.convolve2d(I1M, gx_gaussian, 'same')
    # conv(I, gaussian_y)
    I1My = signal.convolve2d(I1M, gy_gaussian, 'same')

    Ix = (I1x + I2x)/2
    Iy = (I1y + I2y)/2

    IMx = (I1Mx + I2Mx)/2
    IMy = (I1My + I2My)/2
    # plot.imshow(I2x, cmap='gray')
    # plot.show()
    # plot.imshow(I2y, cmap='gray')
    # plot.show()
# round xs and ys to be integers
    newXs = np.zeros(np.shape(startXs))
    newYs = np.zeros(np.shape(startYs))
    for i in range(np.shape(startXs)[1]):
        for j in range(np.shape(startXs)[0]):
            startX = startXs[j, i]
            startY = startYs[j, i]
            newX1, newY1 = estimateFeatureTranslation.estimateFeatureTranslation(startX, startY, Ix, Iy, gray_img1, gray_img2)
            newX2, newY2 = estimateFeatureTranslation.estimateFeatureTranslation(startX, startY, IMx, IMy, I1M, I2M)
            newXs[j, i] = 0.5 * newX1 + 0.5 * newX2
            newYs[j, i] = 0.5 * newY1 + 0.5 * newY2
    #pdb.set_trace()

    return newXs, newYs
