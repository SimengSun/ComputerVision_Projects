import cv2
from cpselect import *
import pdb
from partC_utils import *
from estimateAllTranslation import *

vidcap_source = cv2.VideoCapture('./dataset/target2.mp4')
vidcap_target = cv2.VideoCapture('./dataset/Easy/MrRobot.mp4')

# for i in range(1, 300):
#     success, image_source = vidcap_target.read()


width_source = vidcap_source.get(cv2.CAP_PROP_FRAME_WIDTH)
height_source = vidcap_source.get(cv2.CAP_PROP_FRAME_HEIGHT)

width_target = vidcap_target.get(cv2.CAP_PROP_FRAME_WIDTH)
height_target = vidcap_target.get(cv2.CAP_PROP_FRAME_HEIGHT)

success, image_source_pre = vidcap_source.read()
success, image_target_pre = vidcap_target.read()
image_source_gray_pre = cv2.cvtColor(image_source_pre, cv2.COLOR_BGR2GRAY)
image_target_gray_pre = cv2.cvtColor(image_target_pre, cv2.COLOR_BGR2GRAY)
pts_source, pts_target = cpselect(image_source_pre, image_target_pre)
pdb.set_trace()
pts_source_pre = np.array([[ [587.23686636,  241.97695853]],
       [ [712.68079877,  237.71889401]],
       [ [661.85437788,  282.12442396]],
       [ [613.19078341,  372.76036866]],
       [ [704.02949309,  371.5437788] ]])
pts_target_pre = np.array([[[ 311.80583717,  122.31152074]],
       [ [370.20215054,  124.74470046]],
       [ [339.92258065,  151.81382488]],
       [ [315.0500768 ,  183.44516129]],
       [ [359.92872504,  185.87834101]]])

pts_Xs_source_pre = np.squeeze(pts_source_pre)[:, 0].reshape(5,1)
pts_Ys_source_pre = np.squeeze(pts_source_pre)[:, 1].reshape(5,1)
pts_Xs_target_pre = np.squeeze(pts_target_pre)[:, 0].reshape(5,1)
pts_Ys_target_pre = np.squeeze(pts_target_pre)[:, 1].reshape(5,1)

image_source_gray_pre_res = drawMarks(image_source_gray_pre, np.squeeze(pts_source_pre))
image_target_gray_pre_res = drawMarks(image_target_gray_pre, np.squeeze(pts_target_pre))
cv2.imshow('0', image_source_gray_pre_res)
cv2.waitKey(1000)
for i in range(100):
    success, image_source_cur = vidcap_source.read()
    success, image_target_cur = vidcap_target.read()
    # image_source_gray_cur = cv2.cvtColor(image_source_cur, cv2.COLOR_BGR2GRAY)
    # image_target_gray_cur = cv2.cvtColor(image_target_cur, cv2.COLOR_BGR2GRAY)

    # lk_params = dict(winSize=(10, 10),
    #                  maxLevel=2,
    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    #pts_source_cur, st, err = cv2.calcOpticalFlowPyrLK(image_source_gray_pre, image_source_gray_cur, pts_source_pre.astype('float32'), None, **lk_params)
    #pts_target_cur, st, err = cv2.calcOpticalFlowPyrLK(image_target_gray_pre, image_target_gray_cur, pts_target_pre.astype('float32'), None, **lk_params)

    pts_Xs_source_cur, pts_Ys_source_cur = estimateAllTranslation(pts_Xs_source_pre, pts_Ys_source_pre,
                                                                  image_source_pre, image_source_cur)
    pts_XYs_source_cur = np.concatenate((pts_Xs_source_cur, pts_Ys_source_cur), 1)

    pts_Xs_target_cur, pts_Ys_target_cur = estimateAllTranslation(pts_Xs_target_pre, pts_Ys_target_pre,
                                                                  image_target_pre, image_target_cur)
    pts_XYs_target_cur = np.concatenate((pts_Xs_target_cur, pts_Ys_target_cur), 1)
    pdb.set_trace()
    image_source_cur_res = drawMarks(image_source_cur, pts_XYs_source_cur)
    image_target_cur_res = drawMarks(image_target_cur, pts_XYs_target_cur)

    cv2.imshow(str(i+1), image_source_cur_res)
    cv2.waitKey(1000)

    #pts_source_pre = pts_source_cur
    #pts_target_pre = pts_target_cur

    #image_source_gray_pre = image_source_gray_cur
    #image_target_gray_pre = image_target_gray_cur

    pts_Xs_source_pre = pts_Xs_source_cur
    pts_Ys_source_pre = pts_Ys_source_cur
    pts_Xs_target_pre = pts_Xs_target_cur
    pts_Ys_target_pre = pts_Ys_target_cur
    image_source_pre = image_source_cur
    image_target_pre = image_target_cur