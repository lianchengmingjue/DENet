import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.model_init import *
from torch import channels_last, nn

class ECABlock(nn.Module):
    """Constructs a ECA module.
    
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size

    这个模块应该是另外一篇论文里的，疑似ECANet
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 不清楚是在干嘛，不过看他输入时x，输出也是x，就当是对x的一个增强模块吧，感觉就是基于channel做了一些操作
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        dilation=dilation)


def up_conv3x3(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))




      



class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True,norm=nn.BatchNorm2d, 
        act=F.relu, concat=True,use_att=False, dilations=[], out_fuse=False):
        super(UpConv, self).__init__()
        self.concat = concat
        self.residual = residual
        self.conv2 = []
        self.use_att = use_att
        
        self.out_fuse = out_fuse
        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
            self.norm1 = norm(out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels)
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
        
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down,se=None):
        # from_up分辨率比from_down小一倍，但是channel大一倍
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        
        x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            
            if (se is not None) and (idx == len(self.conv2) - 1): # last 
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        return x2



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d,act=F.relu,residual=True, dilations=[]):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm1 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks
        self.conv2 = []
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
       
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        '''
            返回两个值，卷积结果及其没有下采样之前的特征
        '''
        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool
