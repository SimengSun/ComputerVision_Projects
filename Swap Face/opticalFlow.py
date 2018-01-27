import cv2
from cpselect import *
import pdb
from partC_utils import *
from estimateAllTranslation import *

def opticalFlow(marks_pre, img_pre, img_cur):
    pts_Xs_source_pre = marks_pre[:, 0].reshape(7, 1)
    pts_Ys_source_pre = marks_pre[:, 1].reshape(7, 1)

    pts_Xs_source_cur, pts_Ys_source_cur = estimateAllTranslation(pts_Xs_source_pre, pts_Ys_source_pre,
                                                                  img_pre, img_cur)
    pts_XYs_source_cur = np.concatenate((pts_Xs_source_cur, pts_Ys_source_cur), 1)


    return pts_XYs_source_cur
