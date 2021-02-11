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

class DIFT_NET(nn.Module):
    def __init__(self,args):
        super(DIFT_NET,self).__init__()
    
        self.measurements_length = args["measurements_length"]
        self.dift_code_len = args["dift_code_len"]
        self.keep_prob = 0.9
        self.partition = args["partition"]
        self.dift_code_config = args["dift_code_config"]
        self.training_mode = args["training_mode"]
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        # self.albedo_part = DIFT_NET_ALBEDO(args,args["partition"]["local"],args["dift_code_config"]["local_albedo"][0])
        if self.dift_code_config["local_noalbedo"][0] > 0:
            self.g_diff_local_part = DIFT_NET_G_DIFF_LOCAL(input_size,args["dift_code_config"]["local_noalbedo"][0])
        if self.dift_code_config["global"][0] > 0:
            self.g_diff_global_part = DIFT_NET_G_DIFF_GLOBAL(input_size,args["dift_code_config"]["global"][0])
        # self.g_spec_part = DIFT_NET_G_SPEC(args,args["partition"]["g_spec"][0],args["partition"]["g_spec"][1])
        self.catnet = DIFT_NET_CONCAT(self.dift_code_config)
        
    def forward(self,batch_data,cossin,return_origin_codes=False):
        '''
        batch_data=(batch_size,sample_view_num,m_len,1)
        cossin = (batch_size,2)
        albedo_diff = (batch_size,1)
        albedo_spec = (batch_size,1)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        x_n = batch_data.reshape(batch_size,self.measurements_length)
    
        # dift_codes_albedo = self.albedo_part(x_n[:,m_ptr:m_ptr+self.partition["local"]])

        code_list = []
        origin_codes_map = {}
        if self.dift_code_config["local_noalbedo"][0] > 0:
            dift_codes_g_diff_local = self.g_diff_local_part(x_n,cossin)
            code_list.append(dift_codes_g_diff_local)
            origin_codes_map["local_noalbedo"] = dift_codes_g_diff_local

        if self.dift_code_config["global"][0] > 0:
            dift_codes_g_diff_global = self.g_diff_global_part(x_n,cossin)
            code_list.append(dift_codes_g_diff_global)
            origin_codes_map["global"] = dift_codes_g_diff_global

        if len(code_list) > 1:
            dift_codes = torch.cat(code_list,dim=1)
            dift_codes = self.catnet(dift_codes)
            origin_codes_map["cat"] = dift_codes
        else:
            dift_codes = code_list[0]#
        
        if not return_origin_codes:
            return dift_codes
        else:
            return dift_codes,origin_codes_map