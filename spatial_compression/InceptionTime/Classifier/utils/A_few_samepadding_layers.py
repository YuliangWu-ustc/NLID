import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ShortcutLayer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, use_bias = True):
        super(ShortcutLayer, self).__init__()
        self.use_bias = use_bias
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, bias = self.use_bias)
        self.bn = nn.BatchNorm1d(num_features = out_channels)
    def forward(self, X):
        X = self.padding(X)
        X = F.relu(self.bn(self.conv1d(X)))
        return X    

class SampaddingConv1D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size, use_bias = True):
        super(SampaddingConv1D, self).__init__()
        self.use_bias = use_bias
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels = in_channels, out_channels=out_channels, kernel_size = kernel_size, bias = self.use_bias)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        return X

class SampaddingMaxPool1D(nn.Module):
    def __init__(self,pooling_size, stride):
        super(SampaddingMaxPool1D, self).__init__()
        self.pooling_size = pooling_size
        self.stride = stride
        self.padding = nn.ConstantPad1d((int((pooling_size-1)/2), int(pooling_size/2)), 0)
        self.maxpool1d = nn.MaxPool1d(self.pooling_size, stride=self.stride)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.maxpool1d(X)
        return X