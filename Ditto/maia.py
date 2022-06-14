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
        self.se = SELayer(feature)

    def addmap(self, big, small):
        B, C, H, W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big

    def forward(self, x):
        out = self.m1(x)
        out = self.m2(out)
        out = self.se(out)
        out = self.addmap(x, out)
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

class Maia(nn.Module) :
    def __init__(self,feature , length) -> None:
        super(Maia , self).__init__()
        hid = 64
        self.resenet = Resenet(hid ,length)
        self.downsample = conv2dblock(feature , hid)
        out_hidden = 32
        self.mapout = nn.Sequential(
            nn.ConvTranspose2d(hid, out_hidden, 3),
            nn.BatchNorm2d(out_hidden),
            nn.GELU(),
            nn.ConvTranspose2d(out_hidden, 1, 1),
            nn.Sigmoid()
        )
    def forward(self , x) :
        x = self.downsample(x)
        out = self.resenet(x)
        out = self.mapout(out)
        return out