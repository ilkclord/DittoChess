import glob
import os
import os.path as osp
import sys
from turtle import shape
cwd = os.getcwd().replace("\\train_test", "")
sys.path.append(cwd)


from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchviz import make_dot

import numpy as np

from utils.basic import read_file , getmaxn , getnmax 

class BoardDataset(Dataset):
    def __init__(self, samples , way,usefor = "CNN" ,seq = 1 , shape = "2D"):
        self.data = samples
        self.prev = None
        self.way = way
        self.usefor = usefor
        self.seq = seq
        self.shape = shape
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        if self.way == "bit" : # get 3D
            if self.seq !=  1 :
                raise
            board, from_ans , to_ans  , algrbra= read_file(path , trans = "bit")
        elif self.way == "time-bit" : # get 4D
            board, from_ans, to_ans, algrbra = read_file(path, trans="time-bit" ,seq = self.seq)
        else :
            board, from_ans, to_ans  , _= read_file(path)
        if self.shape == "2D" :  # 13 * 8 * 8
            return torch.reshape(torch.FloatTensor(board) , (-1 ,8 ,8)), torch.FloatTensor(from_ans)
        elif self.shape == "3D" :
            return torch.reshape(torch.FloatTensor(board), (self.seq ,-1, 8, 8)), torch.FloatTensor(from_ans)


# par : target folder , rand SEED

def get_files(folder , SEED):
    train_samples = []
    test_samples = []
    file_list = glob.glob(folder + '/*.txt')
    train_samples, test_samples = train_test_split(file_list, test_size=0.2, random_state=SEED)
    return train_samples, test_samples
def get_files_game(folder , SEED) :
    return 1
# loss functions
from utils.basic import MAGIC
def distance_loss(outputs, targets):
    foutputs = torch.flatten(outputs, start_dim=1).tolist()
    ftargets = torch.flatten(targets, start_dim=1).tolist()
    outputs = outputs.tolist()
    targets = targets.tolist()
    loss = 0
    for idx, output in enumerate(outputs):
        ans = ftargets[idx].index(max(ftargets[idx]))
        maxlist = getmaxn(foutputs[idx])
        try:
            rank = maxlist.index(ans)
            _loss = foutputs[idx][ans] * rank 
            if rank < 3:
                _loss = 0
        except:
            _loss = 7
        loss = loss + _loss 
    return loss/len(outputs)

# Acc function
def accuracy(outputs, targets, maxn):
    """Compute accuracy"""
    #outputs = torch.reshape(outputs,(-1 , 64))
    #targets = torch.reshape(targets,(-1 , 64))
    outputs = torch.flatten(outputs, start_dim=1).tolist()
    targets = torch.flatten(targets, start_dim=1).tolist()
    correct = 0
    ncorrect = []
    for i in range(0, min(len(targets), len(outputs))):
      index = getnmax(outputs[i], maxn)
      #index = outputs[i].index(_index)
      _ans = max(targets[i])
      ans = targets[i].index(_ans)
      if ans in index:
        correct = correct + 1

    return correct / min(len(targets), len(outputs))

# Helper

# For each Epoch
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# Whole train
class recorder(object) :
    def __init__(self , name) :
        self.name = name
        self.logs = []
        self.best = 0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
    def update(self , val , n = 1) :
        self.count += n
        if n == 1 :
            self.logs.append(val)
            self.sum += val
            self.avg = self.sum / self.count
        else :
            self.logs = self.logs + val
            self.sum += sum(val)
            self.avg  = self.sum /self.count
        if val > self.best :
            self.best = val

        
import matplotlib.pyplot as plt
def plot(x, targets, store, ylabel, xlabel="EPOCH", plot_names=[]):
    for target in targets :
        plt.plot(x , target)
    if plot_names != [] :
        if len(plot_names) == len(targets) :
            plt.legend(plot_names)
        else :
            raise SyntaxError("Make sure the plot is 1-1")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(store)
    plt.close()

def store_model(model , filename) :
    torch.save(model.state_dict(), filename)

#train
def train(epoch, model, loader, criterion, optimizer , device , mix , c2 = None):
    """This function trains for 1 epoch and returns the epoch loss and accuracy."""
    # recorders
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Set model to training mode
    model.train()
    for idx, (board, ans) in enumerate(loader):  # load data
        # put the data to correct device
        board = board.to(device)
        ans = ans.to(device)
        # model forward
        outputs = model.forward(board)
        outputs = torch.reshape(outputs, (-1, 8, 8))
        # calculate loss
        loss2 = 0
        if c2 :
            # L1
            loss2 = criterion(torch.reshape(outputs, (-1, 64)),
                              torch.reshape(ans, (-1, 64)))
            # GN
            #loss2 = c2(torch.reshape(outputs, (-1, 64)),torch.reshape(ans, (-1, 64)), torch.ones(outputs.size()[0], 64).to(device))
        loss = criterion(torch.reshape(outputs, (-1, 64)),torch.reshape(ans, (-1, 64)))
        loss =mix[0] * loss + mix[1] * distance_loss(outputs, ans) + mix[2] * loss2
        # clear optimizer gradients
        optimizer.zero_grad()
        # backward to calculate gradients
        loss.backward()
        # update model parameters
        optimizer.step()

        # accuracy
        acc = accuracy(outputs, ans, 1)
        # record loss and acc
        loss_m.update(loss.item(), board.size(0))
        acc_m.update(acc, board.size(0))

        # logging
        print("\rDone {:.3f} %" .format(100*idx/len(loader)), end='')
    return loss_m.avg, acc_m.avg


def test(model, loader, criterion, device, mix , maxns = [1 , 3 , 5 ,7] , c2 = None):
    """This function tests the model on the test set."""
    # recorders
    loss_m = AverageMeter()
    out_list, y_list = [], []
    board_list =[]
    model.eval() # set model to eval mode
    with torch.no_grad(): # no need to compute gradient when testing
        for idx , (board, ans) in enumerate(loader):
            board_list.append(board)
            n = int((board.nelement())/64 )
            #board = torch.reshape(board ,(n ,1 ,8 ,8)).to(device)
            board = board.to(device)
            ans = ans.to(device)
            
            # model forward
            outputs = model.forward(board)
            # calculate loss
            loss2 = 0
            if c2:
                loss2 = criterion(torch.reshape(outputs, (-1, 64)),torch.reshape(ans, (-1, 64)))
                #loss2 = c2(torch.reshape(outputs, (-1, 64)), torch.reshape(ans,(-1, 64)), torch.ones(outputs.size()[0], 64).to(device))
            loss = criterion(torch.reshape(outputs, (-1, 64)),torch.reshape(ans, (-1, 64)))
            loss = mix[0] * loss + mix[1] * distance_loss(outputs, ans) + mix[2] * loss2
            #loss = criterion(torch.reshape(outputs, (-1, 64)), torch.reshape(ans, (-1, 64)), torch.ones(outputs.size()[0], 64).to(device))
            loss = criterion(torch.reshape(outputs, (-1, 64)),torch.reshape(ans, (-1, 64)))
            loss = mix[0] * loss + mix[1] * distance_loss(outputs, ans)
            y_list.append(ans)
            out_list.append(outputs)
            loss_m.update(loss.item())
        # concat all predictions and answers to compute accuracy
        pred = torch.cat(out_list)
        y = torch.cat(y_list)
        acc = accuracy(pred, y , 1)
        acc2 = accuracy(pred , y ,3)
        acc3 =accuracy(pred , y ,5)
        acc4 =accuracy(pred , y ,7)
        #print("\rDone {:.3f} %" .format(100*idx/len(loader)) , end = '')
    
    return loss_m.avg, acc , acc2 ,acc3  ,acc4