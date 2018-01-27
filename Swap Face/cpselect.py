import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

class cpselect_recorder:
	def __init__(self, img1,img2):

		fig, (self.Ax0, self.Ax1) = plt.subplots(1, 2, figsize = (20, 20))

		self.Ax0.imshow(img1)
		self.Ax0.axis('off')

		self.Ax1.imshow(img2)
		self.Ax1.axis('off')

		fig.canvas.mpl_connect('button_press_event', self)
		self.left_x = []
		self.left_y = []
		self.right_x = []
		self.right_y = []

	def __call__(self, event):
		circle = plt.Circle((event.xdata, event.ydata),color='r')
		if event.inaxes == self.Ax0:
			self.left_x.append(event.xdata)
			self.left_y.append(event.ydata)
			self.Ax0.add_artist(circle)
			plt.show()
		elif event.inaxes == self.Ax1:
			self.right_x.append(event.xdata)
			self.right_y.append(event.ydata)
			self.Ax1.add_artist(circle)
			plt.show()

def cpselect(img1,img2):
	resize_img1 = scipy.misc.imresize(img1,[300,300])
	resize_img2 = scipy.misc.imresize(img2,[300,300])
	point = cpselect_recorder(resize_img1,resize_img2)
	plt.show()
	point_left = np.concatenate([(np.array(point.left_x)*img1.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.left_y)*img1.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
	point_right = np.concatenate([(np.array(point.right_x)*img2.shape[1]*1.0/300)[...,np.newaxis],\
								(np.array(point.right_y)*img2.shape[0]*1.0/300)[...,np.newaxis]],axis = 1)
	plt.scatter(point_left[:,0], point_left[:,1])
	plt.imshow(img1)
	plt.show()
	plt.scatter(point_right[:,0], point_right[:,1])
	plt.imshow(img2)
	plt.show()
	return point_left, point_right
