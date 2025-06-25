# this build inception layer that can be calculated parrally.
# the method it uses it using zero mask to on a big convoluation 
# for example: kernel sizes 3 5 7 will be like 
#  0     0    value value value  0     0
#  0    value value value value value  0  
# value value value value value value value  
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def calculate_mask_index(kernel_length_now,largest_kernel_length):
    right_zero_mast_length = math.ceil((largest_kernel_length-1)/2)-math.ceil((kernel_length_now-1)/2)
    left_zero_mask_length = largest_kernel_length - kernel_length_now - right_zero_mast_length
    return left_zero_mask_length, left_zero_mask_length+ kernel_length_now

def creat_mask(number_of_input_channel,number_of_output_channel, kernel_length_now, largest_kernel_length):
    ind_left, ind_right= calculate_mask_index(kernel_length_now,largest_kernel_length)
    mask = np.ones((number_of_input_channel,number_of_output_channel,largest_kernel_length))
    mask[:,:,0:ind_left]=0
    mask[:,:,ind_right:]=0
    return mask

def creak_layer_mask(layer_parameter_list):
    largest_kernel_length = 0
    for layer_parameter in layer_parameter_list:
        if layer_parameter[-1]>largest_kernel_length:
            largest_kernel_length = layer_parameter[-1]
        
        
    mask_list = []
    init_weight_list = []
    bias_list = []
    for i in layer_parameter_list:
        conv = torch.nn.Conv1d(in_channels=i[0], out_channels=i[1], kernel_size=i[2])
        ind_l,ind_r= calculate_mask_index(i[2],largest_kernel_length)
        big_weight = np.zeros((i[1],i[0],largest_kernel_length))
        big_weight[:,:,ind_l:ind_r]= conv.weight.detach().numpy()
        
        bias_list.append(conv.bias.detach().numpy())
        init_weight_list.append(big_weight)
        
        mask = creat_mask(i[1],i[0],i[2], largest_kernel_length)
        mask_list.append(mask)
        
    mask = np.concatenate(mask_list, axis=0)
    init_weight = np.concatenate(init_weight_list, axis=0)
    init_bias = np.concatenate(bias_list, axis=0)
    return mask.astype(np.float32), init_weight.astype(np.float32), init_bias.astype(np.float32)


class Parallel_Inception_Layer(nn.Module):
    # this build inception layer that can be calculated parrally.
    # the method it uses it using zero mask to on a big convoluation 
    # for example: kernel sizes 3 5 7 will be like 
    #  0     0    value value value  0     0
    #  0    value value value value value  0  
    # value value value value value value value  
    
    def __init__(self,layer_parameters, use_bias = True, use_batch_Norm =True, use_relu =True):
        super(Parallel_Inception_Layer, self).__init__()
        
        self.use_bias = use_bias
        self.use_batch_Norm = use_batch_Norm
        self.use_relu = use_relu
        

        os_mask, init_weight, init_bias= creak_layer_mask(layer_parameters)
        
        
        in_channels = os_mask.shape[1] 
        out_channels = os_mask.shape[0] 
        max_kernel_size = os_mask.shape[-1]

        self.weight_mask = nn.Parameter(torch.from_numpy(os_mask),requires_grad=False)
        
        self.padding = nn.ConstantPad1d((int((max_kernel_size-1)/2), int(max_kernel_size/2)), 0)
         
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=max_kernel_size, 
                                      bias =self.use_bias)
        
        self.conv1d.weight = nn.Parameter(torch.from_numpy(init_weight),requires_grad=True)
        self.conv1d.bias =  nn.Parameter(torch.from_numpy(init_bias),requires_grad=True)

        self.bn = nn.BatchNorm1d(num_features=out_channels)
    
    def forward(self, X):
        
        self.conv1d.weight.data = self.conv1d.weight*self.weight_mask
        
        result_1 = self.padding(X)
        result_2 = self.conv1d(result_1)
        
        if self.use_batch_Norm:
            result_3 = self.bn(result_2)
        else:
            result_3 = result_2
            
        if self.use_relu:
            result = F.relu(result_3)
            return result
        else:
            return result_3
