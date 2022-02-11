import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from DIFT_NET_CONCAT import DIFT_NET_CONCAT
from DIFT_NET_G_DIFF_LOCAL import DIFT_NET_G_DIFF_LOCAL
from DIFT_NET_G_DIFF_GLOBAL import DIFT_NET_G_DIFF_GLOBAL
from DIFT_NET_G_SPEC import DIFT_NET_G_SPEC

class DIFT_NET_NORMAL(nn.Module):
    def __init__(self,args):
        super(DIFT_NET_NORMAL,self).__init__()
    
        self.measurements_length = args["measurements_length"]
        self.dift_code_len = args["dift_code_len"]
        self.keep_prob = 0.9
        self.partition = args["partition"]
        self.dift_code_config = args["dift_code_config"]
        self.training_mode = args["training_mode"]
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        self.normalnet = self.normalnet_f(input_size)
        
    def normalnet_f(self,input_size,name_prefix = "NORMALNET_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=512
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=256
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=128
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=64
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_count+=1
        input_size = output_size

        output_size=32
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=3
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
        
    def forward(self,batch_data):
        '''
        batch_data=(batch_size,sample_view_num,m_len,1)
        cossin = (batch_size,2)
        albedo_diff = (batch_size,1)
        albedo_spec = (batch_size,1)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        x_n = batch_data.reshape(batch_size,self.measurements_length)
        
        x_n = torch.nn.functional.normalize(x_n,dim=1)

        x_n = self.normalnet(x_n)

        x_n = torch.nn.functional.normalize(x_n,dim=1)

        return x_n