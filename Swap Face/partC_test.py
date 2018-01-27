import cv2

vidcap = cv2.VideoCapture('./dataset/source.mp4')

width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

success, image1 = vidcap.read()

rec = cv2.selectROI('1',image1)
x,y,w,h = rec[0], rec[1], rec[2], rec[3]
#area = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
image_res = image1
cv2.rectangle(image_res, (x,y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('test', image1)
cv2.waitKey(10000)