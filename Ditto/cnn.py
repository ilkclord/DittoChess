from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchviz import make_dot

from .blocks import afilter , SELayer


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x)


class qModel(nn.Module):
    def __init__(self):
        super(qModel, self).__init__()
        self.re_eval = nn.Sequential(
            torch.nn.Conv2d(1, 32, 1),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU()
        )
        self.measurement = nn.Sequential(
            torch.nn.Conv2d(1, 128, 3),
            torch.nn.Dropout2d(p=0.2),
            nn.GELU(),
            torch.nn.Conv2d(128, 64, 3),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
    def forward(self, x):
        #x = self.re_eval(x)
        bsize = x.size()[0]
        x = self.measurement(torch.reshape(x, (bsize, -1, 8, 8)))
        return x
    def info(self, target):
        print("============ qModel ===========", file=target)
        print("Eval", file=target)
        print("Mesuare", file=target)
        print("Linear", file=target)
        print("Conv->Drop->GELU : 2", file=target)
        print("=====================================", file=target)


class QModel(nn.Module):
    def __init__(self , feature):
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
            torch.nn.BatchNorm2d(64) ,
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
    def addmap(self , big, small) :
        B,C,H,W = big.size()
        return torch.nn.functional.interpolate(small, size=(H, W), mode='bilinear', align_corners=True) + big
    def forward(self, x):
        r1 = self.m1(x)
        r2 = self.m2(r1)
        r3 = self.m3(r2)
        r2 = self.addmap(r1 , r2)
        r1 = self.addmap(r2 , r3)
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
    def __init__(self, blockn , feature):
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
# Test
class QQ(nn.Module):
    def __init__(self , blockn):
        super(QQ, self).__init__()
        self.reeval = nn.Sequential(
            torch.nn.Conv2d(64, 64, 2 , stride=1),
            torch.nn.BatchNorm2d(64),
            nn.GELU(),
            torch.nn.Conv2d(64, 64, 3),
            torch.nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.GELU() ,
            torch.nn.Dropout2d(p=0.5),
            nn.Linear(256, 64),
            torch.nn.Dropout2d(p=0.5),
            nn.Sigmoid()
        )
        self.qs = nn.ModuleList([QModel() for i in range(blockn)])
    def forward(self , x) :
        out = self.qs[0](x)
        for i in range(1, len(self.qs)):
            out += self.qs[i](x)
        out = self.reeval(out)
        out = self.fc(out)
        return out
    def info(self, target):
        print("============ QQ ===========", file=target)
        print("Qmodel : ", len(self.qs), file=target)
        print("=====================================", file=target)   
    
from .blocks import Cnn3GeluBlock , Cnn3ReluBlock
class ImageBlock(nn.Module) :
    def __init__(self , feature ,part = "Net"):
        super(ImageBlock, self).__init__()
        feature = 8
        self.part = part
        self.filter = afilter(feature)
        self.b1 = Cnn3GeluBlock(feature,64, ks = 2)
        self.b2 = Cnn3GeluBlock(64,64)
        self.b3 = Cnn3GeluBlock(64, 64)
        self.fc = nn.Sequential(    
            nn.Linear(576,256) , 
            torch.nn.Dropout2d(p=0.5),
            nn.GELU() ,
            nn.Linear(256 , 64) ,
            nn.GELU()
        )
        self.p1 = nn.AdaptiveMaxPool1d(576)
        self.p2 = nn.AdaptiveMaxPool1d(576)
    def forward(self ,x) :
        if self.part == "Net" :
            x = self.filter(x)
        out1 = self.b1(x)
        out2 = self.b2(out1)
        out3 = self.b3(out2)
        out1 = torch.flatten(out1, start_dim=1)
        out2 = torch.flatten(out2, start_dim=1)
        out3 = torch.flatten(out3, start_dim=1)
        out1 = self.p1(out1)
        out2 = self.p2(out2)
        out = out1 + out2 + out3
        if self.part == "Net" :
            out = torch.flatten(out , start_dim= 1)
            out = self.fc(out)
        return out

    def info(self, target):
        print("============ ImageBlock ===========", file=target)
        print("filter", file=target)
        print("Conv->Drop->GELU : 2", file=target)
        print("Conv->Drop->Sig : 1", file=target)
        print("Linear 576 -> 256 -> 64", file=target)
        print("=====================================", file=target)


class Test(nn.Module):
    def __init__(self, blockn):
        super(Test, self).__init__()
        self.blocks = nn.ModuleList([ImageBlock(1 ,"block") for i in range(blockn)])
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        self.filter = afilter(8)
        self.p1 = nn.AdaptiveAvgPool1d(576)
        self.p2 = nn.AdaptiveAvgPool1d(576)
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = self.filter(x)
        out = self.blocks[0](x)
        for i in range(1 ,len(self.blocks)) :
            out += self.blocks[i](x)
        out = self.fc(out)
        return out
    def info(self, target):
        print("============ Test ===========", file=target)
        print("ImageBlock : ", len(self.blocks), file=target)
        print("Linear 576-> 256-> 128 ->64", file=target)
        print("Residual", file=target)
        print("=====================================", file=target)

# row col seperate
class Test_rcsep(nn.Module):
    def __init__(self, blockn):
        super(Test_rcsep, self).__init__()
        self.blocks = nn.ModuleList(
            [ImageBlock(1, "block") for i in range(blockn)])
        self.rfc = nn.Sequential(
            nn.Linear(576, 256),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 8),
            nn.Tanh()
        )
        self.cfc = nn.Sequential(
            nn.Linear(576, 256),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 8),
            nn.Tanh()
        )
        self.rc = nn.Sequential(
            nn.Linear(16 , 32) ,
            nn.ReLU(),
            nn.Linear(32 , 64) ,
            nn.Tanh()
        )
        self.pb = nn.Sequential(
            nn.Conv2d(1 , 256, 8) ,
            nn.GELU() ,
            nn.Flatten() ,
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        self.rweaken = nn.Sequential(
            nn.PReLU(8) ,
            nn.Sigmoid()
            )
        self.cweaken = nn.Sequential(
            nn.PReLU(8),
            nn.Sigmoid()
        )
        self.filter = afilter(8)
        self.sig = nn.Sigmoid()
        self.p1 = nn.AdaptiveAvgPool1d(576)
        self.p2 = nn.AdaptiveAvgPool1d(576)

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = self.filter(x)
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out += self.blocks[i](x)
        row = self.rfc(out)
        row = self.rweaken(row)
        col = self.cfc(out)
        col = self.cweaken(col)
        row = torch.reshape(col , (-1 , 8 ,1))
        col = torch.reshape(row ,(-1 ,1 ,8))
        out = torch.bmm(row, col)
        #out = self.sig(out)
        #out = torch.reshape(out ,(-1 ,1 ,8,8))
        #out = self.pb(out)
        #out = torch.cat((row , col) ,1)
        #out = self.rc(out)
        return out

    def info(self, target):
        print("============ Test_row_column_seperate ===========", file=target)
        print("ImageBlock : ", len(self.blocks), file=target)
        print("Linear 576-> 256-> 64 ->8 : 2", file=target)
        print("Residual", file=target)
        print("Prelu -> row x col -> Sigmoid", file=target)
        print("Linear 576-> 256-> 64 ->8 : 2", file=target)
        print("=====================================", file=target)
# row col sep block


class rcblock(nn.Module):
    def __init__(self, blockn):
        super(rcblock, self).__init__()
        self.blocks = nn.ModuleList(
            [ImageBlock(1, "block") for i in range(blockn)])
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            torch.nn.Dropout2d(p=0.5),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )
        self.filter = afilter(8)
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = self.filter(x)
        out = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            out += self.blocks[i](x)
        out = self.fc(out)
        return out
    def info(self, target):
        print("============ rcblock ===========", file=target)
        print("ImageBlock : ", len(self.blocks), file=target)
        print("Linear 576-> 256-> 64 ->8 : 2", file=target)
        print("Residual", file=target)
        print("=====================================", file=target)

class Testrc(nn.Module):
    def __init__(self, blockn):
        super(Testrc, self).__init__()
        self.rblock = rcblock(blockn)
        self.cblock = rcblock(blockn)
        self.blockn = blockn
    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        row = self.rblock(x)
        col = self.cblock(x)
        row = torch.reshape(col, (-1, 8, 1))
        col = torch.reshape(row, (-1, 1, 8))
        out = torch.bmm(row, col)
        return out
    def info(self, target):
        print("============ rcblock ===========", file=target)
        print("Rc block with ImageBlock : ", (self.blockn), file=target)
        print("torch bmm ", file=target)
        print("Residual", file=target)
        print("=====================================", file=target)
