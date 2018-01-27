'''
  File name: detectFace.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video
    - Output bbox: the four corners of bounding boxes for all detected faces
'''

import matplotlib.pyplot as plot
import cv2
import numpy as np

def detectFace(img):
  #TODO: Your code here
  rects = []
  num_face = 2
  for i in range(num_face):
    rec = cv2.selectROI("im", img)
    rects.append(rec)
  image_res = img
  for (x, y, w, h) in rects:
    cv2.rectangle(image_res, (x, y), (x + w, y + h), (0, 255, 0), 2)

  cv2.imshow("Face Found", image_res)
  cv2.waitKey(0)

  bbox = []

  for (x, y, w, h) in rects:
    bbox.append([[x,y], [x+w, y], [x, y+h], [x+w, y+h]])

  bbox = np.array(bbox)

  return bbox


