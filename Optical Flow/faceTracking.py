'''
  File name: faceTracking.py
  Author:
  Date created:
'''

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''
import pdb
import detectFace
import getFeatures
import rgb2gray
import estimateAllTranslation
import applyGeometricTransformation
import cv2
import numpy as np
import skimage.feature as ski
import matplotlib.pyplot as plot
from scipy import signal
import skimage.transform as skt

def faceTracking(rawVideo):
  #TODO: Your code here

  vidcap = cv2.VideoCapture(rawVideo)
  width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
  height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

  success, image1 = vidcap.read()
  # success, image1 = vidcap.read()


  image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

  bbox = detectFace.detectFace(image1)

  print bbox

  num_bbox = np.shape(bbox)[0]
  image1Res = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
  startXs, startYs = getFeatures.getFeatures(image1, bbox)
  for i in range(num_bbox):
    cv2.line(image1Res, (bbox[i][0][0], bbox[i][0][1]), (bbox[i][1][0], bbox[i][1][1]), (0, 255, 0), 3)
    cv2.line(image1Res, (bbox[i][0][0], bbox[i][0][1]), (bbox[i][2][0], bbox[i][2][1]), (0, 255, 0), 3)
    cv2.line(image1Res, (bbox[i][1][0], bbox[i][1][1]), (bbox[i][3][0], bbox[i][3][1]), (0, 255, 0), 3)
    cv2.line(image1Res, (bbox[i][3][0], bbox[i][3][1]), (bbox[i][2][0], bbox[i][2][1]), (0, 255, 0), 3)

    image_last = image1

    for j in range(startXs.shape[0]):
      cv2.circle(image1Res, (int(round(startXs[j][i])), int(round(startYs[j][i]))), 1, (255, 0, 0), 2)
  cv2.imshow('Frame 1', image1Res)

  # pdb.set_trace()

  ImageRes_list = []
  for i in range(145):
    success, image_current = vidcap.read()
    print "Frame " + str(i + 2)
    image_current = cv2.cvtColor(image_current, cv2.COLOR_BGR2RGB)
    # remember to interpolate startX and startY(in estimateFeatureTranslatio.py), 'cause from frame 2 to frame 3, there will be xiaoshu
    newXs, newYs = estimateAllTranslation.estimateAllTranslation(startXs, startYs, image_last, image_current)

    # us = newXs - startXs
    # vs = newYs - startYs
    # pdb.set_trace()
    # print newXs-x, newYs-y
    Xs, Ys, newbbox = applyGeometricTransformation.applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
    num_bbox = np.shape(newbbox)[0]
    print newbbox
    # temp = newXs-startXs
    # print "iteration:" +str(i)
    # print temp
    image_currentRes = cv2.cvtColor(image_current, cv2.COLOR_RGB2BGR)
    for k in range(num_bbox):
      cv2.line(image_currentRes, (newbbox[k][0][0], newbbox[k][0][1]), (newbbox[k][1][0], newbbox[k][1][1]),
               (0, 255, 0), 3)
      cv2.line(image_currentRes, (newbbox[k][0][0], newbbox[k][0][1]), (newbbox[k][2][0], newbbox[k][2][1]),
               (0, 255, 0), 3)
      cv2.line(image_currentRes, (newbbox[k][1][0], newbbox[k][1][1]), (newbbox[k][3][0], newbbox[k][3][1]),
               (0, 255, 0), 3)
      cv2.line(image_currentRes, (newbbox[k][3][0], newbbox[k][3][1]), (newbbox[k][2][0], newbbox[k][2][1]),
               (0, 255, 0), 3)
      for j in range(newXs.shape[0]):
        cv2.circle(image_currentRes, (int(round(newXs[j][k])), int(round(newYs[j][k]))), 1, (255, 0, 0), 2)

    ImageRes_list.append(image_currentRes)
    # cv2.imshow('Frame'+ str(i+2), image_currentRes)
    # cv2.waitKey(0)
    print Xs.shape[0]
    if Xs.shape[0] < 15:
      startXs, startYs = getFeatures.getFeatures(image_current, newbbox)
    else:
      startXs = Xs
      startYs = Ys

    image_last = image_current
    bbox = newbbox

  cap = cv2.VideoCapture(0)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  # out = cv2.VideoWriter('output.avi', fourcc, 20, (int(width), int(height)))
  out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (int(width), int(height)))
  for frame in ImageRes_list:
    out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()

  out.release()
  trackedVideo = out


  return trackedVideo