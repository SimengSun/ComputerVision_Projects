import pdb
import cv2
import numpy as np
import skimage.feature as ski
import matplotlib.pyplot as plot
from scipy import signal
import skimage.transform as skt
import PyNet as net
from partC_utils import *
import copy
from opticalFlow import *
from findBbox import *
from Triangulation import *
from getFeatures import *

vidcap_source = cv2.VideoCapture('./dataset/target2.mp4')
vidcap_target = cv2.VideoCapture('./dataset/Easy/MrRobot.mp4')

width_source = vidcap_source.get(cv2.CAP_PROP_FRAME_WIDTH)
height_source = vidcap_source.get(cv2.CAP_PROP_FRAME_HEIGHT)

width_target = vidcap_target.get(cv2.CAP_PROP_FRAME_WIDTH)
height_target = vidcap_target.get(cv2.CAP_PROP_FRAME_HEIGHT)

# # Draw Recs
# success, image = vidcap_target.read()
# rec = cv2.selectROI("im", image)
# image_res = image
# x,y,w,h = rec[0],rec[1],rec[2],rec[3]
# #x,y,w,h = 370, 0, 500, 500
# cv2.rectangle(image_res, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# area = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
# print x, y, w, h
# cv2.imshow("Face Found", image_res)
# cv2.waitKey(0)

'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''
layer_list = [
                net.Conv2d(output_channel=64, kernel_size=5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.Conv2d(output_channel=64, kernel_size=5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.MaxPool2d(kernel_size=2, stride=2, padding=0),
                net.Conv2d(output_channel=128, kernel_size=5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.Conv2d(output_channel=128, kernel_size=5, padding=2, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.MaxPool2d(kernel_size=2, stride=2, padding=0),
                net.Conv2d(output_channel=384, kernel_size=3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.Conv2d(output_channel=384, kernel_size=3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Relu(),
                net.Conv2d(output_channel=5, kernel_size=3, padding=1, stride=1),
                net.BatchNorm2D(),
                net.Sigmoid(),
                net.Upsample(size=(40, 40))
             ]
'''
  Define loss function
'''
loss_layer = net.Binary_cross_entropy_loss(average=True)

'''
  Define optimizer 
'''
lr = 1e-4
wd = 5e-4
mm = 0.99
optimizer = net.SGD_Optimizer(lr_rate=lr, weight_decay=wd, momentum=mm)

'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer)

my_model.set_input_channel(3)

'''
  Input possible pre-trained model
'''
my_model.load_model('final.pickle')


x0_source = []
x1_source = []
x2_source = []
x3_source = []
x4_source = []
y0_source = []
y1_source = []
y2_source = []
y3_source = []
y4_source = []

# For optical flow
pts_source_pre = np.array([[ 626.85591398,  244.440553  ],
       [ 743.64854071,  251.13179724],
       [ 675.51950845,  304.66175115],
       [ 632.26298003,  368.53271889],
       [ 718.77603687,  369.74930876],
       [ 597.24856205, 249.46350439 ],
       [ 766.17536426, 258.41478875 ]])
pts_target_pre = np.array([[ 427.51705069,  131.59410138],
       [ 506.46021505,  138.28534562],
       [ 463.74439324,  162.31299539],
       [ 426.97634409,  206.11023041],
       [ 489.15760369,  212.49732719],
       [ 408.0516129 ,  135.8521659 ],
       [ 530.79201229,  143.76      ]])


# 1st Frame
success, image_source_pre = vidcap_source.read()
success, image_target_pre = vidcap_target.read()


ImageRes_list = []

for i in range(1, 130):
    success, image_source = vidcap_source.read()
    success, image_target = vidcap_target.read()
    image_source_cur = copy.deepcopy(image_source)
    image_target_cur = copy.deepcopy(image_target)
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB).astype('float64')
    image_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2RGB).astype('float64')

    #inds = getFeatures(image_source, )

    pts_source_cur = opticalFlow(pts_source_pre, image_source_pre, image_source_cur)
    pts_target_cur = opticalFlow(pts_target_pre, image_target_pre, image_target_cur)

    # image_source_gray_pre = cv2.cvtColor(image_source_pre, cv2.COLOR_BGR2GRAY)
    # image_target_gray_pre = cv2.cvtColor(image_target_pre, cv2.COLOR_BGR2GRAY)
    # image_source_gray_cur = cv2.cvtColor(image_source_cur, cv2.COLOR_BGR2GRAY)
    # image_target_gray_cur = cv2.cvtColor(image_target_cur, cv2.COLOR_BGR2GRAY)
    #
    # lk_params = dict(winSize=(10, 10),
    #                  maxLevel=2,
    #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # pts_source_cur, st, err = cv2.calcOpticalFlowPyrLK(image_source_gray_pre, image_source_gray_cur,
    #                                                    pts_source_pre.astype('float32'), None, **lk_params)
    # pts_target_cur, st, err = cv2.calcOpticalFlowPyrLK(image_target_gray_pre, image_target_gray_cur,
    #                                                    pts_target_pre.astype('float32'), None, **lk_params)

    # Update Prev
    pts_source_pre = pts_source_cur
    pts_target_pre = pts_target_cur
    image_source_pre = image_source_cur
    image_target_pre = image_target_cur

# low resolution
    # source
    # image_res_source = copy.deepcopy(image_source)
    x_s, y_s, w_s, h_s = 460, 70, 440, 440
    area_s = [[x_s, y_s], [x_s + w_s, y_s], [x_s, y_s + h_s], [x_s + w_s, y_s + h_s]]
    # target
    # image_res_target = copy.deepcopy(image_target)
    x_t, y_t, w_t, h_t = 362, 14, 278, 260
    area_t = [[x_t, y_t], [x_t + w_t, y_t], [x_t, y_t + h_t], [x_t + w_t, y_t + h_t]]

    img_area_source = copy.deepcopy(image_source[y_s:y_s+h_s, x_s:x_s+w_s, :])
    img_area_target = copy.deepcopy(image_target[y_t:y_t+h_t, x_t:x_t+w_t, :])

    ori_img_source, img_source = preprocess(img_area_source)
    ori_img_target, img_target = preprocess(img_area_target)

    pred_source = my_model.forward(img_source)
    pred_target = my_model.forward(img_target)

    marks_source = getLandMarks(pred_source, x_s, y_s, h_s, w_s)
    marks_target = getLandMarks(pred_target, x_t, y_t, h_t, w_t)

# Lerp CNN results and Optical Flow results
    marks_source = marks_source * 0.5 + pts_source_cur[0:5] * 0.5
    marks_target = marks_target * 0.5 + pts_target_cur[0:5] * 0.5

    other_marks_source = pts_source_cur[5:7]
    other_marks_target = pts_target_cur[5:7]

    marks_source_bbox = np.concatenate((marks_source, other_marks_source), 0)
    marks_target_bbox = np.concatenate((marks_target, other_marks_target), 0)
    #img_res_source = drawMarks(image_source, marks_source)
    #img_res_target = drawMarks(image_target, marks_target)

# # Update Optical Flow pts using average results
#     pts_source_pre = copy.deepcopy(marks_source)
#     pts_target_pre = copy.deepcopy(marks_target)
#
#     temp = copy.deepcopy(pts_source_pre[:, 0])
#     pts_source_pre[:, 0] = pts_source_pre[:, 1]
#     pts_source_pre[:, 1] = temp
#     temp = copy.deepcopy(pts_target_pre[:, 0])
#     pts_target_pre[:, 0] = pts_target_pre[:, 1]
#     pts_target_pre[:, 1] = temp

# Find BBox
    minX_source, maxX_source, minY_source, maxY_source = findBbox(marks_source_bbox, 12, 60, 50, height_source, width_source)
    minX_target, maxX_target, minY_target, maxY_target = findBbox(marks_target_bbox, 15, 40, 40, height_target, width_target)

    #img_res_source = drawBbox(img_res_source, minX_source, maxX_source, minY_source, maxY_source)
    #img_res_target = drawBbox(img_res_target, minX_target, maxX_target, minY_target, maxY_target)

# Get Face Box Image
    img_face_source = copy.deepcopy(image_source[minY_source:maxY_source, minX_source:maxX_source, :])
    img_face_target = copy.deepcopy(image_target[minY_target:maxY_target, minX_target:maxX_target, :])

    offset_pts_source = getOffsetPts(marks_source, minX_source, minY_source)
    offset_pts_target = getOffsetPts(marks_target, minX_target, minY_target)

## Warp And Morph

# resize source to target
    img_face_source = scipy.misc.imresize(img_face_source, [maxY_target-minY_target, maxX_target-minX_target])
    offset_pts_source[:,0] = offset_pts_source[:,0] * float(maxX_target-minX_target) / float(maxX_source-minX_source)
    offset_pts_source[:,1] = offset_pts_source[:,1] * float(maxY_target-minY_target) / float(maxY_source-minY_source)

    img_face_source_res = drawMarks(img_face_source, offset_pts_source)
    img_face_target_res = drawMarks(img_face_target, offset_pts_target)

    # cv2.imshow('source', img_face_source_res.astype('uint8'))
    # cv2.waitKey(1000)
    # cv2.imshow('target', img_face_target_res.astype('uint8'))
    # cv2.waitKey(0)

    all_mark_pts_source = getAllMarks(offset_pts_source, minX_target, maxX_target, minY_target, maxY_target)
    all_mark_pts_target = getAllMarks(offset_pts_target, minX_target, maxX_target, minY_target, maxY_target)

    tri = triangulation(all_mark_pts_target)

# generate meshgrid
    nx, ny = (maxX_target - minX_target, maxY_target - minY_target)
    xx = np.linspace(0, nx-1, nx)
    yy = np.linspace(0, ny-1, ny)

    xv, yv = np.meshgrid(xx, yy)
    # y first
    xyv = np.array([xv, yv]).transpose(1,2,0)

    simplex = tri.find_simplex(xyv)

    xyv_inverse = tri.transform[simplex, :2]
    xyv_vec = (xyv - tri.transform[simplex, 2])
    xyv_bary = copy.deepcopy(xyv)

    new_img_face_source = copy.deepcopy(img_face_source)

    for m in range(maxX_target - minX_target):
        for n in range(maxY_target - minY_target):
            xy_bary = np.matmul(xyv_inverse[n, m], xyv_vec[n, m])
            xyv_bary = xy_bary
            indices = tri.simplices[simplex[n, m]]
            # x first
            xy_source = all_mark_pts_source[indices[0]] * xy_bary[0] + all_mark_pts_source[indices[1]] * xy_bary[1] + all_mark_pts_source[indices[2]] * (1.0 - xy_bary[0] - xy_bary[1])
            new_img_face_source[n, m, :] = img_face_source[int((xy_source[1])), int((xy_source[0])), :]

    src_mask = np.ones(new_img_face_source.shape, new_img_face_source.dtype)
    src_mask *= 255
    poly = np.array([[0,0], [0, new_img_face_source.shape[1]], [new_img_face_source.shape[0], 0], [new_img_face_source.shape[0], new_img_face_source.shape[1]]], np.int32)
    center = (minX_target + int((maxX_target - minX_target)/2.0), minY_target + int((maxY_target - minY_target)/2.0))
    img_res = cv2.seamlessClone(new_img_face_source.astype('uint8'), image_target.astype('uint8'), src_mask, center, cv2.NORMAL_CLONE)

    img_res = drawBbox(img_res, minX_target, maxX_target, minY_target, maxY_target)
    # img_res = copy.deepcopy(image_target)
    # img_res[minY_target:maxY_target, minX_target:maxX_target] = new_img_face_source
    cv2.imshow(str(i+1), cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR).astype('uint8'))
    cv2.waitKey(1)
    # cv2.imshow(str(i+10000), img_res.astype('uint8'))
    # cv2.waitKey(0)
    #plot.figure()
    #plot.imshow(img_res_target.astype('uint8'))
    #plot.show()
    #pdb.set_trace()
    #pdb.set_trace()
    print 'frame: ' + str(i)
    x0_source.append(marks_source[0][0])
    x1_source.append(marks_source[1][0])
    x2_source.append(marks_source[2][0])
    x3_source.append(marks_source[3][0])
    x4_source.append(marks_source[4][0])

    # cv2.rectangle(image_res_source, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 255, 0), 2)
    #
    # cv2.imshow("Face Found", image_res_source)
    # cv2.waitKey(500)

    ImageRes_list.append(cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))

# plot.plot(range(len(x0_source)), x0_source)
# plot.show()
# plot.plot(range(len(x1_source)), x1_source)
# plot.show()
# plot.plot(range(len(x2_source)), x2_source)
# plot.show()
# plot.plot(range(len(x3_source)), x3_source)
# plot.show()
# plot.plot(range(len(x4_source)), x4_source)
# plot.show()


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20, (int(width), int(height)))
out = cv2.VideoWriter('outpy_1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (int(width_target), int(height_target)))
for frame in ImageRes_list:
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

out.release()