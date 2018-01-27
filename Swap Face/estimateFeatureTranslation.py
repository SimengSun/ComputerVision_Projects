import numpy as np
import cv2
import pdb
import scipy.io
from scipy import interpolate
import matplotlib.pyplot as plot
'''
  File name: estimateFeatureTranslation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for single features 
    - Input startX: the x coordinate for single feature wrt the first frame
    - Input startY: the y coordinate for single feature wrt the first frame
    - Input Ix: the gradient along the x direction
    - Input Iy: the gradient along the y direction
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newX: the x coordinate for the feature wrt the second frame
    - Output newY: the y coordinate for the feature wrt the second frame
'''

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    window_size = 3

    # padding zeros
    gray_img1 = np.pad(img1, window_size, 'constant', constant_values=0)
    gray_img2 = np.pad(img2, window_size, 'constant', constant_values=0)

    Ix = np.pad(Ix, window_size, 'constant', constant_values=0)
    Iy = np.pad(Iy, window_size, 'constant', constant_values=0)

    # scipy.io.savemat('data.mat', mdict={'gray_img1':gray_img1,'gray_img2':gray_img2,'I2x':I2x,'I2y':I2y})


    # TODO: Your code here
    x_c = int(startX + window_size)
    y_c = int(startY + window_size)

    # get windows 11x11
    I1_current_window = gray_img1[-window_size + y_c:window_size+1 + y_c, -window_size + x_c:window_size+1 + x_c]
    I2_current_window = gray_img2[-window_size + y_c:window_size+1 + y_c, -window_size + x_c:window_size+1 + x_c]
    Ix_current_window = Ix[-window_size + y_c:window_size+1 + y_c, -window_size + x_c:window_size+1 + x_c]
    Iy_current_window = Iy[-window_size + y_c:window_size+1 + y_c, -window_size + x_c:window_size+1 + x_c]
    It = I2_current_window - I1_current_window

    # xx, yy = np.meshgrid(range(11), range(11))
    # # interpolate 11x11
    # I1_interp = interpolate.interp2d(xx, yy, I1_current_window, kind='cubic')
    # I2_interp = interpolate.interp2d(xx, yy, I2_current_window, kind='cubic')
    # It_interp = interpolate.interp2d(xx, yy, It, kind='cubic')
    #
    # xv, yv = np.linspace(0, 9, 10), np.linspace(0, 9, 10)
    # xv = xv + startX - int(startX)
    # yv = yv + startY - int(startY)
    # I1_current_window = I1_interp(xv, yv)
    # I2_current_window = I2_interp(xv, yv)
    #
    # It = It_interp(xv, yv)

    # plot.imshow(I1_current_window, cmap='gray')
    # plot.show()
    # plot.imshow(I2_current_window, cmap='gray')
    # plot.show()
    # plot.imshow(I2x_current_window, cmap='gray')
    # plot.show()
    # plot.imshow(I2y_current_window, cmap='gray')
    # plot.show()
    # plot.imshow(It)
    # plot.show()

    Jacobian_left = np.zeros([2, 2])
    Jacobian_right = np.zeros([2, 1])

    for m in range(window_size*2+1):
        for n in range(window_size*2+1):
            Jacobian_left[0, 0] += Ix_current_window[m, n] * Ix_current_window[m, n]
            Jacobian_left[0, 1] += Ix_current_window[m, n] * Iy_current_window[m, n]
            Jacobian_left[1, 0] = Jacobian_left[0, 1]
            Jacobian_left[1, 1] += Iy_current_window[m, n] * Iy_current_window[m, n]
            Jacobian_right[0, 0] += Ix_current_window[m, n] * It[m, n]
            Jacobian_right[1, 0] += Iy_current_window[m, n] * It[m, n]

    #pdb.set_trace()
    inv_Jacobian_left = np.linalg.inv(Jacobian_left)
    uv = np.matmul(inv_Jacobian_left, -Jacobian_right)

    newX = startX + uv[0]
    newY = startY + uv[1]
    #pdb.set_trace()

    # cull by borders
    # nr, nc = gray_img1.shape
    # if newX >= nr or newX < 0 or newY >= nc or newY < 0:
    #     newX = -1
    #     newY = -1
    return newX, newY


