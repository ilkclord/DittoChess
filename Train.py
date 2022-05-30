from matplotlib.pyplot import show
from Ditto.cnn import QQQ, QQ
from Ditto.timecnn import detect
from Ditto.test import model_test
from Ditto.lstm import Memoryblock
from time import time
from sklearn.datasets import load_boston
from Ditto.blocks import afilter
from train_test.training import plot
from train_test.training import recorder
from train_test.training import get_files, BoardDataset
from train_test.training import test, train
from torch.utils.data import DataLoader
import os
import sys
cwd = os.getcwd().replace("\\train_test", "")
sys.path.append(cwd)

import torch
# Hypermeters pass by argv
# python exe model_name source visualize? epochs? pretrained?
model_name = sys.argv[1]
source = sys.argv[2]
visual = ""
try :
    visual = sys.argv[3]
except :
    visual = ""
EPOCHS = 0
try : 
    EPOCHS = int(sys.argv[4])
except :
    EPOCHS = 0
# keep training 
load_path = ""
try :
    load_path = sys.argv[5]
except :
    load_path = ""
# Data
BATCH_SIZE = 32
SEED = 17
USE_CUDA = 1
# Loader
timeseq = 2
way = "time-bit"
shape = "3D"

# Model
OPTIM = "Adam"
LOSS = "CE"


# Train 
# ce dist gauss
mix = [0.8 , 0.2 , 0.0]
LR = 1e-3
device = 'cpu'
if USE_CUDA and torch.cuda.is_available():  # 若想使用 cuda 且可以使用 cuda
    device = 'cuda'
print(f'Using {device} for training.')

# Get target Model
#from Ditto.timecnn import Timemodel
#target_model = Memoryblock(seq = lstm_seq , sig= lstm_sig , lstmn=lstm_layern ,types = lstm_type)
#target_model = QQQ(3, timeseq *13)
from Ditto.timecnn import detect_Model , Timemodel , detect_test , TT , TT_2
from Ditto.deconvTT import DCTT
#target_model = detect_Model(timeseq, 3 , 3)
#target_model = Timemodel(timeseq)
#target_model = QQQ(3, timeseq * 13)
#target_model = detect_test(timeseq)
target_model = DCTT(timeseq , 1)
if shape == "2D":
    model_test(target_model, torch.randn(BATCH_SIZE, timeseq * 14, 8, 8))
else:
    model_test(target_model, torch.randn(BATCH_SIZE, timeseq , 14 , 8, 8))
if load_path != "":
    target_model.load_state_dict(torch.load(load_path))
# Get Source Data
train_sp , test_sp = get_files(source , SEED)

# Create Datasets
train_dataset = BoardDataset(train_sp , way , seq = timeseq ,shape = shape)
test_dataset = BoardDataset(test_sp , way , seq = timeseq , shape = shape)

# Create Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE ,shuffle=True, num_workers=0)
print(f'There are {len(train_loader)} training batches and {len(test_loader)} testing batches')


# Prepare loss , optimizer
optimizer = None
if OPTIM == "Adam" :
    optimizer = torch.optim.Adam(target_model.parameters(), lr=LR, betas=(0.9, 0.99))
elif OPTIM == "SGD" :
    optimizer = torch.optim.SGD(target_model.parameters(), lr=0.01, momentum=0.5)
else :
    print("None")
next_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.01, momentum=0.5)
import torch.nn as nn
criterion = None
if LOSS == "MSE" :
    criterion = nn.MSELoss()
elif LOSS == "CE" :
    criterion = nn.CrossEntropyLoss()
elif LOSS == "GN" :
    criterion = nn.GaussianNLLLoss()
else:
    print("None")
GN = nn.GaussianNLLLoss()
L1 = nn.L1Loss()
L2 = nn.MSELoss()
l2 = L1
# Get Epoch
if EPOCHS == 0 :
    EPOCHS= int(input("EPOCH : "))
SEED = 17
# Start training 
test_acc = recorder("Accuracy1")
test_acc3 = recorder("Accuracy3")
test_acc5 = recorder("Accuracy5")
test_acc7 = recorder("Accuracy7")
train_acc = recorder("Train")
train_loss = recorder("Train")
test_loss = recorder("Test")
best_acc = (0, 0.0)  # (epoch, acc)
# Put model into device
target_model=  target_model.to(device)
for epoch in range(EPOCHS):
    print("---------------- Epoch {} ----------------".format(epoch+1))

    # training
    loss, acc = train(epoch, target_model, train_loader,criterion, optimizer, device, mix , c2 = l2)
    # saving epoch loss and acc for plotting
    train_loss.update(loss)
    train_acc.update(acc)
    # logging
    print('\nEpoch {:3d}/{} Train Loss {:.3f} Acc {:.3f}'.format(epoch+1, EPOCHS, loss, acc))

    # testing
    print("---------------- Testing ----------------")
    loss, acc, acc3, acc5, acc7 = test(target_model, test_loader, criterion, device, mix , c2 = l2)
    test_loss.update(loss)
    test_acc.update(acc)
    test_acc3.update(acc3)
    test_acc5.update(acc5)
    test_acc7.update(acc7)
    # logging
    print('Epoch {:3d}/{} Test Loss {:.3f} Acc {:.3f} ,ACc {:.3f} ,ACC {:.3f} ,ACCC {:.3f}'.format(
        epoch+1, EPOCHS, loss, acc, acc3, acc5, acc7))

print('The Best accuracy is {:.3f} '.format(test_acc.best))
#record
RECORD_PATH = "train_test/ModelZoo/" + model_name
if not os.path.isdir(RECORD_PATH):
    os.mkdir(RECORD_PATH)
RECORD_PATH = RECORD_PATH + '/'
torch.save(target_model.state_dict(), RECORD_PATH + 'model.path')
x = list(range(EPOCHS))
plot(x , [train_acc.logs , test_acc.logs , test_acc3.logs , test_acc5.logs] , RECORD_PATH + "Accuracy" , "Accuracy",
         plot_names=[train_acc.name , test_acc.name , test_acc3.name , test_acc5.name ])
plot(x , [train_loss.logs , test_loss.logs] ,RECORD_PATH + "Loss" , "Loss")

file = open(RECORD_PATH + "logs.txt", 'w')
print("Train Data : ", source, file=file)
print("shape : {:} , timeseq : {:} , way : {:}".format(shape , timeseq , way) , file = file)
print("Epoch : ", EPOCHS, file=file)
print("Structure : " , file =file)
try :
    target_model.info(file)
except :
    print("Lazy.." , file = file)
print("Loss : {:.2f} * {:} + {:.2f} * Distance Loss".format(mix[0] ,LOSS, mix[1]) ,file = file)
print("Optimizer : " , OPTIM ,file = file)
print("===================================" , file = file)
print('Train loss : {:.3f} '.format(min(train_loss.logs)), file=file)
print('Test loss : {:.3f} '.format(min(test_loss.logs)), file = file)
print('Train best accuracy is {:.3f} '.format(train_acc.best) , file = file)
print('Test top1 best accuracy is {:.3f} '.format(test_acc.best), file=file)
print('Test top3 best accuracy is {:.3f} '.format(test_acc3.best), file=file)
file.close()
print("Done training for ", model_name, " with", source)

from utils.basic import read_file
from utils.visualizer import draw_board_score
import cv2
if visual != "" :
    from utils.visualizer import visualizer
    shown = 0
    if visual != "all" :
        shown = int(visual)
        print("Visualized ", shown, " testing data")
        visualizer(target_model, RECORD_PATH,test_sp[SEED:SEED + shown], device=device, type=way, seq=timeseq , shape = shape)
    else :
        visualizer(target_model, RECORD_PATH, test_sp, device=device,type= way , seq = timeseq)

