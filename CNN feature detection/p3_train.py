'''
  File name: p3_train.py
  Author:
  Date:
'''

import PyNet as net
from p3_utils import *
from p3_dataloader import *
import matplotlib.pyplot as plot

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

'''
  Define the number of input channel and initialize the model
'''
my_model.set_input_channel(3)


'''
  Input possible pre-trained model
'''
# my_model.load_model('preMolde.pickle')


'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data

img_lst, label_lst = load_list()
#img_lst, label_lst = load_test_list()

max_epoch_num = 5
batch_size = len(img_lst)
mini_batch_size = 1
step = batch_size / mini_batch_size
save_interval = 1

'''
  pre-process data
  1. normalization
  2. convert ground truth data 
  3. resize data into the same size
'''
Loss_List = []
Accuracy_List = []

for i in range (max_epoch_num):
  '''
    random shuffle data 
  '''
  img_lst_cur, label_lst_cur = randomShuffle(img_lst, label_lst) # design function by yourself

  total_accuracy_offset = 0
  total_loss = 0
  # step = ...  # step is a int number
  for j in range (step):
    # obtain a mini batch for this step training
    [img_lst_bt, label_lst_bt] = obtainMiniBatch(img_lst_cur, label_lst_cur, j, mini_batch_size)  # design function by yourself

    ori_data_bt, data_bt, label_bt = preprocess(img_lst_bt, label_lst_bt)

    # feedward data and label to the model
    loss, pred = my_model.forward(data_bt, label_bt)
    accuracy_offset = getAccuracyOffset(pred, label_bt)
    total_accuracy_offset += accuracy_offset
    total_loss += loss
    # backward loss
    my_model.backward(loss)

    # update parameters in model
    my_model.update_param()

    print "  Step: " + str(j)
    print "    Loss: " + str(loss)
    print "    Accuracy: " + str(accuracy_offset)

  print "iteration: " + str(i)
  print "  Loss: " + str(total_loss)
  print "  Accuracy: " + str(total_accuracy_offset)
  Loss_List.append(total_loss)
  Accuracy_List.append(total_accuracy_offset)
  '''
    save trained model, the model should be saved as a pickle file
  '''
  if i % save_interval == 0:
    my_model.save_model(str(i) + '.pickle')

plot.plot(range(len(Loss_List)), Loss_List)
plot.show()
plot.plot(range(len(Loss_List)), Accuracy_List)
plot.show()