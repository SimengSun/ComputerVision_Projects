
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

my_model.set_input_channel(3)

'''
  Input possible pre-trained model
'''
my_model.load_model('final.pickle')


'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''


img_lst, label_lst = load_test_list()

total_accuracy_offset = 0
total_loss = 0

for i in range (0, len(img_lst)):
    '''
    random shuffle data 
    '''
    #img_lst_cur, label_lst_cur = randomShuffle(img_lst, label_lst) # design function by yourself

    # obtain a mini batch for this step training
    [img_lst_bt, label_lst_bt] = obtainMiniBatch(img_lst, label_lst, i, 1)  # design function by yourself

    ori_data_bt, data_bt, label_bt = preprocess(img_lst_bt, label_lst_bt)

    # feedward data and label to the model
    loss, pred = my_model.forward(data_bt, label_bt)


    accuracy_offset = getAccuracyOffset(pred, label_bt)
    total_accuracy_offset += accuracy_offset

    # update parameters in model
    #my_model.update_param()
    total_loss += loss
    print "Image Test: " + str(i)
    print "Loss: " + str(loss)
    print "Accuracy Offset: " + str(accuracy_offset)

print "Average Loss: " + str(total_loss/len(img_lst))
print "Average Accuracy Offset" + str(total_accuracy_offset/len(img_lst))