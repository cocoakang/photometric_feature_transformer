import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
import mine

class ALBEDO_NET(nn.Module):
    def __init__(self,args):
        super(ALBEDO_NET,self).__init__()
    
        self.measurements_length = args["measurements_length_albedo"]
        self.keep_prob = 0.9
        self.share_code_len = 256
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        self.albedo_net_share = self.albedo_net_share_f(input_size)
        self.diffuse_net = self.diffuse_net_f(self.share_code_len)
        self.specular_net = self.specular_net_f(self.share_code_len)

    def albedo_net_share_f(self,input_size,name_prefix = "ALBEDO_NET_SHARE_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=128
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=256
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=256
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=self.share_code_len
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def diffuse_net_f(self,input_size,name_prefix = "DIFFUSE_NET_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=256
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=128
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
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

        output_size=32
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=1
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"Sigmoid_{}".format(layer_count)] = nn.Sigmoid()
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
    
    def specular_net_f(self,input_size,name_prefix = "SPECULAR_NET_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=256
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=128
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
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

        output_size=32
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=1
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"Sigmoid_{}".format(layer_count)] = nn.Sigmoid()
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def forward(self,batch_data):
        '''
        batch_data=(batch_size,sample_view_num,m_len,1)
        return:
        albedo_diff=(batch_size,1)
        albedo_spec=(batch_size,1)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        x_n = batch_data.reshape(batch_size,self.measurements_length)

        share_codes = self.albedo_net_share(x_n)
        
        albedo_diff = self.diffuse_net(share_codes)*(mine.param_bounds["pd"][1]-mine.param_bounds["pd"][0])+mine.param_bounds["pd"][0]
        albedo_spec = self.specular_net(share_codes)*(mine.param_bounds["ps"][1]-mine.param_bounds["ps"][0])+mine.param_bounds["ps"][0]

        return albedo_diff,albedo_spec