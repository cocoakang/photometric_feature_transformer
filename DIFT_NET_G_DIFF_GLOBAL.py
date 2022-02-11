import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math

class DIFT_NET_G_DIFF_GLOBAL(nn.Module):
    def __init__(self,measurements_length,dift_code_len):
        super(DIFT_NET_G_DIFF_GLOBAL,self).__init__()
    
        self.measurements_length = measurements_length
        self.dift_code_len = dift_code_len
        self.internal_code_len = 8
        self.keep_prob = 0.9
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        self.dift_part = self.dift_part_f(input_size)
        self.dift_part2 = self.dift_part_f2(self.internal_code_len+9+3)

    def dift_part_f(self,input_size,name_prefix = "DIFT_"):
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

        # output_size=512
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=1024
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=2048
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=1024
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_count+=1
        # input_size = output_size

        # output_size=512
        # # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        # layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        # layer_count+=1
        # input_size = output_size

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

        output_size=self.internal_code_len
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack
    
    def dift_part_f2(self,input_size,name_prefix = "DIFT2_"):
        layer_stack = OrderedDict()
        
        layer_count = 0

        output_size=max(self.dift_code_len,256)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=max(self.dift_code_len,512)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=max(self.dift_code_len,512)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=max(self.dift_code_len,256)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        output_size=max(self.dift_code_len,128)
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_stack[name_prefix+"LeakyRelu_{}".format(layer_count)] = nn.LeakyReLU(negative_slope=0.2)
        layer_count+=1
        input_size = output_size

        # assert self.dift_code_len % 2 == 0
        output_size=self.dift_code_len#//2
        # layer_stack[name_prefix+"BN_{}".format(layer_count)] = nn.BatchNorm1d(input_size)
        # layer_stack[name_prefix+"Dropout_{}".format(layer_count)] = nn.Dropout(1-self.keep_prob)
        layer_stack[name_prefix+"Linear_{}".format(layer_count)] = nn.Linear(input_size,output_size)
        layer_count+=1
        input_size = output_size

        layer_stack = nn.Sequential(layer_stack)

        return layer_stack

    def forward(self,batch_data,cossin):
        '''
        batch_data=(batch_size,sample_view_num,m_len,1)
        view_ids_cossin = (batch_size,2)
        view_mat_model_t = (batch_size,16)
        view_mat_for_normal_t = (batch_size,16)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        # view_codes = self.view_part(torch.zeros_like(view_ids_cossin))

        x_n = batch_data.reshape(batch_size,self.measurements_length)
        # print(x_n)
        x_n = torch.nn.functional.normalize(x_n,dim=1)
        # print(x_n)
        # x_n = x_n.reshape(batch_size,-1)
        # x_n = torch.cat([x_n,view_codes],dim=1)

        dift_codes = self.dift_part(x_n)
        # dift_codes = torch.nn.functional.normalize(dift_codes,dim=1)

        # dift_codes_pos = torch.cat([dift_codes,view_mat_model_t],dim=1)
        dift_codes_normal = torch.cat([dift_codes,cossin],dim=1)
        # dift_codes_pos = nn.functional.sigmoid(self.dift_part2(dift_codes_pos))
        dift_codes_normal = self.dift_part2(dift_codes_normal)
        dift_codes_normal = torch.nn.functional.normalize(dift_codes_normal,dim=1)

        dift_codes = dift_codes_normal#torch.cat([dift_codes_pos,dift_codes_normal],dim=1)
        # print(dift_codes)
        # exit()

        return dift_codes#,torch.zeros(batch_size,3,dtype=torch.float32)