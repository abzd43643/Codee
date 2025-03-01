import torch
import torch.nn.functional as F
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, 2, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
        
class CBAM(nn.Module):
    def __init__(self, dim,kernel_size=7):
        super(CBAM, self).__init__()
        self.sa = SpatialAttention(kernel_size)
        self.ca = ChannelAttention(dim)
    def forward(self, x):
        x = self.sa(x)*x
        x = self.ca(x)*x
        return x 

class SEB(nn.Module):
    def __init__(self, dim=256,kernel_size=7):
        super().__init__()
        self.cbam0 = CBAM(dim,kernel_size)
        self.cbam1 = CBAM(dim,kernel_size)
        self.cbam2 = CBAM(dim,kernel_size)
    def forward(self, x, bins):
        bins = F.interpolate(input=bins, size=(x.size()[2], x.size()[3]), mode='bilinear', align_corners=True).bool()
        bin0 = bins[:, 0, :, :].unsqueeze(1)
        bin1 = bins[:, 1, :, :].unsqueeze(1)
        bin2 = bins[:, 2, :, :].unsqueeze(1)
        x0 = bin0*x
        x1 = bin1*x
        x2 = bin2*x
        
        x0 = self.cbam0(x0)
        x1 = self.cbam1(x1)
        x2 = self.cbam2(x2)
        x = (x0 + x1 +x2 ) + x
        return x
        
class SEM(nn.Module):
    def __init__(self,dim=256):
        super().__init__()
        self.se1 = SEB(dim,9)
        self.se2 = SEB(dim*2,7)
        self.se3 = SEB(dim*4,5)
        self.se4 = SEB(dim*8,3)
    def forward(self, feats, bin):
        x1 = self.se1(feats[0], bin)
        x2 = self.se2(feats[1], bin)
        x3 = self.se3(feats[2], bin)
        x4 = self.se4(feats[3], bin)
        return x1, x2, x3, x4        