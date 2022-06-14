import torch
import torch.nn as nn

# for algebratic input
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x , start_dim= 1)
class afilter(nn.Module) :
    def __init__(self , sig ) :
        super(afilter, self).__init__()
        self.re_eval = nn.Sequential(
            torch.nn.Conv2d(1 ,sig , 1) ,
            torch.nn.Dropout2d(p = 0.5) ,
            nn.Tanh()
            )
        self.flat = nn.Flatten()
        self.combine = nn.Sequential(
        	nn.Linear(sig * 64 , 256) ,
        	nn.GELU() ,
        	nn.Linear(256 , 64) ,
        	nn.GELU()
        	)
    def forward(self ,x) :
        #bsize = x.size()[0]
        x = self.re_eval(x)
        #x = self.flat(x)
        #x = self.combine(x)
        return x 
    def info(self ,file) :
        print("I am filter ?" ,file = file)
class Cnn3GeluBlock(nn.Module) :
    def __init__(self, inn, out, padding=0 , ks = 3):
        super(Cnn3GeluBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inn, out, ks, padding = padding),
            nn.GELU() ,
            torch.nn.Dropout2d(p=0.5)
        )
    def forward(self ,x) :
        return self.block(x)


class Cnn3ReluBlock(nn.Module):
    def __init__(self, inn, out, padding=0, ks=3):
        super(Cnn3ReluBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inn, out, ks, padding=padding),
            nn.ReLU(),
            torch.nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        return self.block(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c,1, 1, 1)
        return x * y.expand_as(x)
