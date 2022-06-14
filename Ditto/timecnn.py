from turtle import forward
from cv2 import merge
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .blocks import  SELayer3D  ,SELayer
from .test import model_test
class detect(nn.Module) :
    def __init__(self , seq , len , stride = 1):
        super(detect , self).__init__()
        self.m1 = nn.Sequential(
            nn.Conv3d(seq , 64 ,(len , 3,3) , stride = stride) ,
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        self.m2 = nn.Sequential(
            nn.Conv3d(64, 64, (len, 3, 3), stride=stride),
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        self.m3 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=stride),
            nn.BatchNorm3d(64),
            nn.GELU()
        )
        self.se = SELayer3D(64)
        self.flat = nn.Flatten()
        f = 1792
        self.fc = nn.Sequential(
            nn.Linear(f, int(f/2)),
            nn.GELU(),
            nn.Linear(int(f/2), 64),
            nn.Sigmoid()
        )
    def forward(self ,x) :
        out = self.m1(x)
        out = self.m2(out)
        out = self.m3(out)
        out = self.se(out)
        out = self.flat(out)
        #out = self.fc(out)
        return out
class detect_Model(nn.Module) :
    def __init__(self , seq ,len ,blockn):
        super(detect_Model , self).__init__()
        self.dts = nn.ModuleList([detect(seq , len) for i in range(blockn)])
        f = 1792
        self.fc = nn.Sequential(
            nn.Linear(f, int(f/2)),
            nn.GELU(),
            nn.Linear(int(f/2), 64),
            nn.Sigmoid()
        )
    def forward(self , x) :
        out = self.dts[0](x)
        for i in range(len(self.dts)) :
            if i == 0 :
                continue 
            out += self.dts[i](x) 
        out = self.fc(out)
        return out
class Timemodel(nn.Module) :
    def __init__(self , seq ):
        super(Timemodel , self).__init__()
        self.t1 = detect(seq , 2)
        self.t2 = detect(seq , 3)
        self.t3 = detect(seq , 5)
        self.seq = seq
        f = 4864
        f2 = int(f/2)
        f3 = int(f2/2)
        self.fc = nn.Sequential(
            nn.Linear(f, f2),
            nn.GELU(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(f2 , f3),
            nn.GELU() ,
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(f3 , 64) ,
            nn.Sigmoid()
            #nn.Softmax(dim = 1)
        )
    def merge(self ,first ,targets) :
        for t in targets :
            first = torch.cat((first , t) , dim = 1)
        return first

    def addmap(self, big, small):
        B, C, H, L ,W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big
    def addmerge(self, big, targets):
        for t in targets:
            print("fasdf")
    def info(self, target):
        print("============ TTT ===========", file=target)
        print("Time_seq : ", self.seq, file=target)
        print("Detect Model -> 2 , 3 , 4", file=target)
        print("=====================================", file=target)
    def forward(self, x):
        o1 = self.t1(x)
        o2 = self.t2(x)
        o3 = self.t3(x)
        out = self.merge(o1 , [o2 , o3])
        out = self.fc(out)
        return out

"""
            nn.Conv3d(hidden, hidden, (3,3 , 3), stride=stride),
            nn.BatchNorm3d(hidden),
            nn.GELU() ,
"""
class element_wise(nn.Module) :
    def __init__(self , shape , seq):
        super(element_wise , self).__init__()
        self.weights = nn.parameter.Parameter(torch.randn(shape))
        self.through = nn.Sequential(
            nn.BatchNorm3d(seq),
            nn.GELU()
        )
    def forward(self ,x) :
        return self.through(x * self.weights)

class detect_test(nn.Module):
    def __init__(self, seq, mode = "Net" ,stride=1):
        super(detect_test, self).__init__()
        self.ochannel = 64
        ochannel = 64 
        hidden = 128
        self.hidden = hidden
        out = 64
        #self.e1 = element_wise((1, 13,8,8) ,seq)
        self.m1 = nn.Sequential(
            nn.Conv3d(seq,hidden, (5, 1, 1), stride=stride),
            nn.BatchNorm3d(hidden),
            nn.GELU() ,
            nn.Conv3d(hidden, ochannel, (9, 3, 3),stride=stride),       
            nn.BatchNorm3d(ochannel),
            nn.GELU()
        )
        self.m2 = nn.Sequential(
            torch.nn.Conv2d(ochannel, ochannel, 3),
            torch.nn.BatchNorm2d(ochannel),
            nn.GELU()
        )
        self.m3 = nn.Sequential(
            torch.nn.Conv2d(ochannel, ochannel, 3),
            torch.nn.BatchNorm2d(ochannel),
            nn.GELU()
        )
        self.m4 = nn.Sequential(
            torch.nn.Conv2d(ochannel, ochannel, 7),
            torch.nn.BatchNorm2d(ochannel),
            nn.GELU()
        )
        self.mode = mode
        self.se = SELayer(out)
        self.flat = nn.Flatten()
        f = 2304
        f2 = int(f/2)
        f3 = int(f2/2)
        self.fc = nn.Sequential(
            nn.Linear(f, f2),
            nn.GELU(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(f2, 64),
            nn.Sigmoid()
            #nn.Softmax(dim = 1)
        )

    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big
    def forward(self, x):
        #x = self.e1(x)
        out = self.m1(x)
        r1 = torch.reshape(out , (-1 , self.ochannel, 6, 6))
        r2 = self.m2(r1)
        r3 = self.m3(r2)
        #r4 = self.m4(r3)
        #r2 = self.addmap(r1, r2)
        #r1 = self.addmap(r2, r3)
        r1 = self.addmap(r2 , r3)
        r1 = self.se(r1)
        out = self.flat(r1)
        if self.mode == "Net" :
            out = self.fc(r1)
        return out
class TT(nn.Module) :
    def __init__(self, seq, blockn ,stride=1):
        super(TT, self).__init__()
        self.seq = seq
        self.dts = nn.ModuleList([detect_test(seq , mode = "layer") for i in range(blockn)])
        f = 1024
        f2 = int(f/2)
        f3 = int(f2/2)
        self.fc = nn.Sequential(
            nn.Linear(f, f2),
            nn.Sigmoid(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(f2, 64),
            nn.Sigmoid()
            #nn.Softmax(dim = 1)
        )
    def info(self, target):
        print("============ TT===========", file=target)
        print("Time_seq : ", self.seq, file=target)
        print("Detect Model 2-3D -> 3 , 2 , 2", file=target)
        print("DM number : " , len(self.dts) , file=target)
        print("=====================================", file=target)
    def forward(self , x) :
        out = self.dts[0](x)
        for i in range(len(self.dts)) :
            if i == 0 :
                continue 
            out += self.dts[i](x) 
        out = self.fc(out)
        return out
class detect2D(nn.Module) :
    def __init__(self ,feature):
        super(detect2D, self).__init__()
        self.m1 = nn.Sequential(
            torch.nn.Conv2d(feature, feature, 3),
            torch.nn.BatchNorm2d(feature),
            nn.GELU()
        )
        self.m2 = nn.Sequential(
            torch.nn.Conv2d(feature, feature, 3),
            torch.nn.BatchNorm2d(feature),
            nn.GELU()
        )
        self.m3 = nn.Sequential(
            torch.nn.Conv2d(feature, feature, 3),
            torch.nn.BatchNorm2d(feature),
            nn.GELU()
        )
        self.se = SELayer(feature)
        self.flat = nn.Flatten()
    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big
    def forward(self ,x) :
        f1  = self.m1(x)
        f2 = self.m2(f1)
        f3 = self.m3(f2)
        #out = self.addmap(f1 , f2)
        #out = self.addmap(out , f3)
        out = f3
        out = self.se(out)
        return out

class TT_2(nn.Module):
    def __init__(self, seq, blockn, stride=1):
        super(TT_2, self).__init__()
        self.seq = seq
        ochannel = 64
        self.ochannel = ochannel
        hidden = 64
        mapn = 64
        k1 = 1
        k2 = 3
        h1 = 14
        h2 = 15 - h1
        f = 3136
        self.filter = nn.Sequential(
            nn.Conv3d(seq, hidden, (h1, k1, k1), stride=stride),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, ochannel, (h2, k2, k2), stride=stride),
            nn.BatchNorm3d(ochannel),
            nn.GELU()
        )
        
        self.mapping = nn.Sequential(
            nn.ConvTranspose2d(64 , mapn ,4) ,
            torch.nn.BatchNorm2d(mapn),
            nn.GELU()
        )
        self.dts = nn.ModuleList(
            [detect2D(mapn) for i in range(blockn)]) 
        self.flat = nn.Flatten()
        f2 = int(f/2)
        f3 = int(f2/2)
        out_hidden = 64
        self.mapout = nn.Sequential(
            nn.ConvTranspose2d(64, out_hidden, 5),
            torch.nn.BatchNorm2d(out_hidden),
            nn.GELU(),
            nn.ConvTranspose2d(out_hidden, 1, 2),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(f, f2),
            nn.Sigmoid(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(f2, 64),
            nn.Sigmoid()
            #nn.Softmax(dim = 1)
        )

    def info(self, target):
        print("============ TT===========", file=target)
        print("Time_seq : ", self.seq, file=target)
        print("Detect Model 2-3D -> 3 , 2 , 2", file=target)
        print("DM number : ", len(self.dts), file=target)
        print("=====================================", file=target)

    def forward(self, x):
        x = self.filter(x)
        x = torch.squeeze(x, dim = 2)
        x = self.mapping(x)
        out = self.dts[0](x)
        for i in range(len(self.dts)):
            if i == 0:
                continue
            out += self.dts[i](x)
        #out = self.flat(out)
        #out = self.fc(out)
        out = self.mapout(out)
        return out
#model_test(Timemodel(3), torch.randn(32, 3, 13, 8, 8))