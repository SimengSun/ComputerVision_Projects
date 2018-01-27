'''
  File name: p3_utils.py
  Author:
  Date:
'''
import numpy as np
import scipy.ndimage as scimg
import math

def randomShuffle(data, label):
    '''
    :param data: shuffled image list
    :param label: shuffled label list
    :return:
    '''
    shuffle_indx = np.random.permutation(len(data))
    new_data = []
    new_label = []
    for i in range(len(shuffle_indx)):
        new_data.append(data[shuffle_indx[i]])
        new_label.append(label[shuffle_indx[i]][:10])
    return new_data, new_label
    #return data, label

def obtainMiniBatch(data, label, step, mini_batch_size):
    '''
    :param data:
    :param label:
    :param step:
    :return:
    '''
    if (step+1) * mini_batch_size > len(data):
        return data[step*mini_batch_size:], label[step*mini_batch_size:]
    else:
        return data[step*mini_batch_size:(step+1)*mini_batch_size], \
               label[step*mini_batch_size:(step+1)*mini_batch_size]

def getAccuracyOffset(pred, labels):
    total_dis = 0;
    for i in range(labels.shape[1]):
        maxid_label = np.unravel_index(np.argmax(labels[0][i]), (40, 40))
        pred[0][i][0:3][0:3] = 0
        pred[0][i][0:3][37:40] = 0
        pred[0][i][37:40][37:40] = 0
        pred[0][i][37:40][0:40] = 0
        maxid_pred = np.unravel_index(np.argmax(pred[0][i]), (40, 40))
        dis = math.sqrt((maxid_label[0] - maxid_pred[0])**2 + (maxid_label[1]-maxid_pred[1])**2)
        total_dis += dis

    return total_dis