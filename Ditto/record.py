from turtle import forward
from cv2 import merge
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .blocks import SELayer3D, SELayer
from .test import model_test
class detect_test(nn.Module):
    def __init__(self, seq, mode="Net", stride=1):
        super(detect_test, self).__init__()
        self.ochannel = 64
        ochannel = 64
        hidden = 128
        self.m1 = nn.Sequential(
            nn.Conv3d(seq, hidden, (5, 1, 1), stride=stride),
            nn.BatchNorm3d(hidden),
            nn.GELU(),
            nn.Conv3d(hidden, ochannel, (9, 3, 3), stride=stride),
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
        self.mode = mode
        self.se = SELayer(ochannel)
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
        out = self.m1(x)
        r1 = torch.reshape(out, (-1, self.ochannel, 6, 6))
        r2 = self.m2(r1)
        r3 = self.m3(r2)
        r2 = self.addmap(r1, r2)
        r1 = self.addmap(r2, r3)
        r1 = self.se(r1)
        out = self.flat(r1)
        if self.mode == "Net":
            out = self.fc(r1)
        return out


class TT(nn.Module):
    def __init__(self, seq, blockn, stride=1):
        super(TT, self).__init__()
        self.seq = seq
        self.dts = nn.ModuleList(
            [detect_test(seq, mode="layer") for i in range(blockn)])
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
        print("DM number : ", len(self.dts), file=target)
        print("=====================================", file=target)

    def forward(self, x):
        out = self.dts[0](x)
        for i in range(len(self.dts)):
            if i == 0:
                continue
            out += self.dts[i](x)
        out = self.fc(out)
        return out

class QModel(nn.Module):
    def __init__(self, feature):
        super(QModel, self).__init__()
        self.m1 = nn.Sequential(
            torch.nn.Conv2d(feature, 64, 3),
            torch.nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.m2 = nn.Sequential(
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.m3 = nn.Sequential(
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.se = SELayer(64)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.Sigmoid(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(256, 64),
            nn.Sigmoid()
        )

    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big

    def forward(self, x):
        r1 = self.m1(x)
        r2 = self.m2(r1)
        r3 = self.m3(r2)
        r2 = self.addmap(r1, r2)
        r1 = self.addmap(r2, r3)
        r1 = self.se(r1)
        #out = self.fc(r1)
        return r1

    def info(self, target):
        print("============ qModel ===========", file=target)
        print("Eval", file=target)
        print("Mesuare", file=target)
        print("Linear", file=target)
        print("Conv->Drop->GELU : 2", file=target)
        print("=====================================", file=target)

# Best
class QQQ(nn.Module):
    def __init__(self, blockn, feature):
        super(QQQ, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 256),
            nn.Sigmoid(),
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(256, 64),
            nn.Sigmoid()
            #nn.Softmax(dim = 1) # sig 2 soft
        )
        self.qs = nn.ModuleList([QModel(feature) for i in range(blockn)])

    def forward(self, x):
        out = self.qs[0](x)
        for i in range(1, len(self.qs)):
            out += self.qs[i](x)
        out = self.fc(out)
        return out

    def info(self, target):
        print("============ QQQ ===========", file=target)
        print("Qmodel : ", len(self.qs), file=target)
        print("=====================================", file=target)
