# Plate_Recognition-LPRnet
this code is based on the paper :      LPRNet: License Plate Recognition via Deep Neural Networks
 https://arxiv.org/pdf/1806.10447.pdf
 
 it is a kind of light network for plate recognition.
 
 it uses CNN + CTC loss to recognise the plate without segmentation.
 
 
Step 1: set the default value

In the file LPRtf3.py：

num_epochs = 300

INITIAL_LEARNING_RATE = 1e-3

DECAY_STEPS = 2000

LEARNING_RATE_DECAY_FACTOR = 0.9 # The learning rate decay factor

MOMENTUM = 0.9

REPORT_STEPS = 5000

#the num of training data

BATCH_SIZE = 50

TRAIN_SIZE = 7368

BATCHES = TRAIN_SIZE//BATCH_SIZE

test_num = 3

ti = 'train' #the location of training data

vi = 'valid' #the location of validation data

img_size = [94, 24]

tl = None

vl = None

num_channels = 3 

label_len = 7 #the length of plate character 


Step 2: start training

running 

$python3 LPRtf3.py

and then the screen will show:

'train or test:'

then:

input 'train' for training

input 'test' for testing


if you want to train your own data,you just need to rename your plate file as "province(convert the Chinese character  to the coresponding code according to 'dict' in the LPRnet.py)_the numbers and alphabet of the plate", the examples is in the folder 'train'. 
