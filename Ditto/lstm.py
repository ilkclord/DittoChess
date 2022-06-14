import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .blocks import afilter

import numpy as np
class Memoryblock(nn.Module) :
	def __init__(self , seq , sig , lstmn , types) :
		super(Memoryblock, self).__init__()
		self.seq = seq
		self.fsig = sig
		self.layern = lstmn
		self.types = "bit"
		self.memory = nn.LSTM(64 , 128, lstmn, dropout = 0.5 , batch_first = True)
		#self.prev = torch.reshape(torch.FloatTensor(np.zeros(64)) , (1,-1)).to('cuda')
		if types == "ab" :
			self.filter = afilter(sig)
			self.types = types
		self.flat = nn.Flatten()
		self.linear = nn.Sequential(
			nn.Linear(seq * 128 , 64) , 
			nn.Sigmoid()
			)
		self.tranf = None
	def forward(self ,x) :
		bsize = x.size()[0]
		if self.types == "ab" :
			x = torch.reshape(x , (-1, 1 , 8 , 8))
			x = self.filter(x)
		x = torch.reshape(x , (bsize , -1 , 64))
		x ,hc = self.memory(x)
		x = self.flat(x)
		x = self.linear(x)
		x = torch.reshape(x , (-1 , 8 , 8))
		return x
	def info(self , target) :
		print("============ Memory Block ===========" , file = target)
		print("LSTM depth : " , self.layern , file = target)
		print("Tranform Value Map : " , self.fsig , file = target)
		print("Time seq : " , self.seq , file = target)
		print("=====================================" , file=target)
		