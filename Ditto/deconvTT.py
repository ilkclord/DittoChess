from re import L
from turtle import forward
from cv2 import merge
from matplotlib.pyplot import cla
from numpy import block
from soupsieve import select
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .blocks import SELayer3D, SELayer




class detect2D(nn.Module):
    def __init__(self, feature):
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
        self.m4 = nn.Sequential(
            torch.nn.Conv2d(feature, feature, 3),
            torch.nn.BatchNorm2d(feature),
            nn.GELU()
        )
        self.m5 = nn.Sequential(
            torch.nn.Conv2d(feature, feature, 3),
            torch.nn.BatchNorm2d(feature),
            nn.GELU()
        )
        self.se = SELayer(feature)
        self.flat = nn.Flatten()

    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big

    def forward(self, x):
        f1 = self.m1(x)
        f2 = self.m2(f1)
        f3 = self.m3(f2)
        f4 = self.m4(f3)
        f5 = self.m5(f4)
        #out = self.addmap(f1 , f2)
        #out = self.addmap(out , f3)
        out = self.addmap(f3, f4)
        out = self.addmap(out, f5)
        #out = self.addmap(f4 , f5)
        #out = f3
        out = self.se(out)
        return out




class DCTT(nn.Module):
    def __init__(self, seq, blockn, stride=1):
        super(DCTT, self).__init__()
        self.seq = seq
        ochannel = 64
        self.ochannel = ochannel
        hidden = 64
        mapn = 64
        k1 = 1
        k2 = 3
        h1 = 14
        h2 = 15 - h1
        self.filter = nn.Sequential(
            nn.Conv3d(seq, hidden, (h1, k1, k1), stride=stride),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, ochannel, (h2, k2, k2), stride=stride),
            nn.BatchNorm3d(ochannel),
            nn.GELU()
        )
        maphidden = 32
        self.mapping = nn.Sequential(
            nn.ConvTranspose2d(64,maphidden, 6),
            torch.nn.BatchNorm2d(maphidden),
            nn.GELU(),
            nn.ConvTranspose2d(maphidden, mapn, 2),
            torch.nn.BatchNorm2d(mapn),
            nn.GELU()
        )
        self.dts = nn.ModuleList(
            [detect2D(mapn) for i in range(blockn)])
        out_hidden = 32
        self.mapout = nn.Sequential(
            nn.ConvTranspose2d(mapn, out_hidden, 3),
            nn.BatchNorm2d(out_hidden) ,
            nn.GELU() ,
            nn.ConvTranspose2d(out_hidden, 1, 1),
            nn.Sigmoid()
        )

    def info(self, target):
        print("============ TT===========", file=target)
        print("Time_seq : ", self.seq, file=target)
        print("Detect Model 2-3D -> 3 , 2 , 2", file=target)
        print("DM number : ", len(self.dts), file=target)
        print("=====================================", file=target)

    def forward(self, x):
        x = self.filter(x)
        x = torch.squeeze(x, dim=2)
        x = self.mapping(x)
        #print(x.size())
        out = self.dts[0](x)
        for i in range(len(self.dts)):
            if i == 0:
                continue
            out += self.dts[i](x)
        #print(out.size())
        out = self.mapout(out)
        #print(out.size())
        return out
