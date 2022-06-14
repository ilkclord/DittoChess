import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .deconvTT import detect2D
from .blocks import SELayer3D, SELayer
def conv2dblock(feature, out, k=3):
    return nn.Sequential(
        torch.nn.Conv2d(feature, out, k),
        torch.nn.BatchNorm2d(out),
        nn.GELU()
    )


class residual_block(nn.Module):
    def __init__(self, feature, type="block"):
        super(residual_block, self).__init__()
        hidden = 64
        self.type = type
        self.m1 = conv2dblock(feature, hidden)
        self.m2 = conv2dblock(hidden, hidden)
        self.m3 = conv2dblock(hidden, hidden)
        self.m4 = conv2dblock(hidden, feature)
        if self.type != "block":
            self.m5 = conv2dblock(feature, feature)
        self.se = SELayer(feature)

    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big

    def forward(self, x):
        out = self.m1(x)
        out = self.m2(out)
        out = self.m3(out)
        out = self.m4(out)
        out = self.se(out)
        if self.type == "block":
            out = self.addmap(x, out)
            return out
        else:
            out = self.m5(out)
            return out


class Resenet(nn.Module):
    def __init__(self, feature, blockn):
        super(Resenet, self).__init__()
        self.rbs = nn.ModuleList([residual_block(feature)
                                 for i in range(blockn)])

    def forward(self, x):
        out = self.rbs[0](x)
        for i in range(len(self.rbs)):
            if i == 0:
                continue
            out = self.rbs[i](out)
        return out

class Ditto_test(nn.Module):
    def __init__(self, seq, blockn, stride=1):
        super(Ditto_test, self).__init__()
        self.seq = seq
        ochannel = 64
        self.ochannel = ochannel
        self.blockn = blockn
        self.dmap = torch.tensor
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
            nn.ConvTranspose2d(64, maphidden, 4),
            torch.nn.BatchNorm2d(maphidden),
            nn.GELU(),
            nn.ConvTranspose2d(maphidden, mapn, 4),
            torch.nn.BatchNorm2d(mapn),
            nn.Sigmoid()
        )
        self.resenet = Resenet(mapn, blockn)
        self.fetch = detect2D(mapn)
        out_hidden = 32
        self.mapout = nn.Sequential(
            nn.ConvTranspose2d(mapn, out_hidden, 3),
            nn.BatchNorm2d(out_hidden),
            nn.GELU(),
            nn.ConvTranspose2d(out_hidden, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.filter(x)
        x = torch.squeeze(x, dim=2)
        out = self.mapping(x)
        if self.training != True :
            self.dmap = out
        out = self.resenet(out)
        out = self.fetch(out)
        out = self.mapout(out)
        return out

    def info(self, target):
        print("============ Ditto Test ===========", file=target)
        print("Time_seq : ", self.seq, file=target)
        print("Map -> Resenet -> Map", file=target)
        print("Resenet length : ", self.blockn, file=target)
        print("=====================================", file=target)
