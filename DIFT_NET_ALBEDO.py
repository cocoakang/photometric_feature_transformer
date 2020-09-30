import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class DIFT_NET_ALBEDO(nn.Module):
    def __init__(self,args,measurements_length,dift_code_len):
        super(DIFT_NET_ALBEDO,self).__init__()
    
        self.measurements_length = measurements_length
        self.dift_code_len = dift_code_len
        self.keep_prob = 0.9
        #############construct model
        input_size = self.measurements_length*1
        
        self.dift_albedo = self.dift_albedo_f(input_size)
    
    def dift_albedo_f(self,input_size,name_prefix = "DIFTALBEDO__"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=32
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=64
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=128
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=128
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=64
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=32
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=self.dift_code_len
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def forward(self,batch_data):
        '''
        batch_data=(batch_size,2)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        dift_codes = self.dift_albedo(batch_data)
        
        dift_codes = torch.nn.functional.normalize(dift_codes,dim=1)

        return dift_codes