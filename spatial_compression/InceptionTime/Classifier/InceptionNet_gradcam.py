import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .utils.Parallel_Inception_Layer import Parallel_Inception_Layer
from .utils.A_few_samepadding_layers import ShortcutLayer, SampaddingConv1D, SampaddingMaxPool1D

class Inception_module(nn.Module):
    def __init__(self, input_channle_size, nb_filters, bottleneck_size, kernel_sizes, stride = 1, activation = 'linear'):
        super(Inception_module, self).__init__()
        self.input_channle_size = input_channle_size
        self.nb_filters = nb_filters
        self.bottleneck_size = bottleneck_size
        self.kernel_sizes = kernel_sizes-1
        self.stride = stride
        self.activation = activation 
        
        self.n_incepiton_scale = 3
        self.kernel_size_s = [self.kernel_sizes // (2 ** i) for i in range(self.n_incepiton_scale)]
        
        if self.input_channle_size > 1 and self.bottleneck_size!= None:
            self.bottleneck_layer = SampaddingConv1D(self.input_channle_size, self.bottleneck_size,kernel_size = 1, use_bias = False)
            self.layer_parameter_list = [ (self.bottleneck_size,self.nb_filters ,kernel_size) for kernel_size in self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list,use_bias = False, use_batch_Norm = False, use_relu =False)                
        else:
            self.layer_parameter_list = [ (self.input_channle_size,self.nb_filters ,kernel_size) for kernel_size in self.kernel_size_s]
            self.parallel_inception_layer = Parallel_Inception_Layer(self.layer_parameter_list,use_bias = False, use_batch_Norm = False, use_relu =False)
        
            
        self.maxpooling_layer = SampaddingMaxPool1D(3,self.stride)
        self.conv_6_layer = SampaddingConv1D(self.input_channle_size,self.nb_filters, kernel_size = 1, use_bias = False)
        
        self.output_channel_numebr = self.nb_filters*(self.n_incepiton_scale+1)
        self.bn_layer = nn.BatchNorm1d(num_features=self.output_channel_numebr)
        
        
        self.grad = []
        self.features = []
        if self.input_channle_size > 1 and self.bottleneck_size!= None:
            self.bottleneck_layer.register_forward_hook(self.save_features_hook) # 注册hook
        
    def save_features_hook(self, module, input, output):
        self.features.append(output)
        # print(f"Features: {output.shape}")
        
    def gradient_hook(self, grad):
        # print(f"Gradients: {grad.shape}")
        self.grad.append(grad)
        
    def forward(self,X):
        if X.shape[-2] >1:
            input_inception = self.bottleneck_layer(X)
        else: 
            input_inception = X
            
        if X.shape[-2] >1:
            input_inception.register_hook(self.gradient_hook)
        
        concatenateed_conv_list_result  = self.parallel_inception_layer(input_inception)
        conv_6 = self.conv_6_layer(self.maxpooling_layer(X))
        
        
        
        concatenateed_conv_list_result_2 = torch.cat((concatenateed_conv_list_result,conv_6),1)
        result = F.relu(self.bn_layer(concatenateed_conv_list_result_2))
        return result
        
        
        
class InceptionNet(nn.Module):
    def __init__(self, 
                 input_channle_size, 
                 nb_classes, 
                 verbose=False, 
                 build=True, 
                 nb_filters=32, 
                 use_residual=True, 
                 use_bottleneck=True, 
                 depth=6, 
                 kernel_size=41):
        super(InceptionNet, self).__init__()
        
        self.input_channle_size = input_channle_size
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        if use_bottleneck:
            self.bottleneck_size = 32
        else:
            self.bottleneck_size = None
            
    
        self.res_layer_list = nn.ModuleList()
        self.layer_list = nn.ModuleList()
        self.out_put_channle_number_list = []
        
        for d in range(self.depth):
            if d == 0:
                input_channle_size_for_this_layer = self.input_channle_size
            else:
                input_channle_size_for_this_layer = self.out_put_channle_number_list[-1]
            inceptiontime_layer = Inception_module(input_channle_size_for_this_layer, 
                             self.nb_filters, 
                             self.bottleneck_size, 
                             self.kernel_size,
                             stride = 1, 
                             activation = 'linear')
            self.layer_list.append(inceptiontime_layer)
            self.out_put_channle_number_list.append(inceptiontime_layer.output_channel_numebr)

            if self.use_residual and d % 3 == 2:
                if d ==2:
                    shortcutlayer = ShortcutLayer(self.input_channle_size, self.out_put_channle_number_list[-1], kernel_size = 1, use_bias = False)
                else:   
                    shortcutlayer = ShortcutLayer(self.out_put_channle_number_list[-4], self.out_put_channle_number_list[-1], kernel_size = 1, use_bias = False)
                self.res_layer_list.append(shortcutlayer)
        
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(self.out_put_channle_number_list[-1], self.nb_classes)
    
    def forward(self, X):
        res_layer_index = 0
        input_res = X
        for d in range(self.depth):
            X = self.layer_list[d](X)
            if self.use_residual and d % 3 == 2:
                shot_cut = self.res_layer_list[res_layer_index](input_res)
                res_layer_index = res_layer_index + 1
                X = torch.add(shot_cut,X)
                input_res = X
                
        X = self.averagepool(X)
        X = X.squeeze_(-1)
        X = self.hidden(X)
        return X
            
        
    