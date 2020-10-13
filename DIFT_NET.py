import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from DIFT_NET_ALBEDO import DIFT_NET_ALBEDO
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
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        self.albedo_part = DIFT_NET_ALBEDO(args,args["partition"]["local"],args["dift_code_config"]["local_albedo"][0])
        self.g_diff_local_part = DIFT_NET_G_DIFF_LOCAL(args,args["partition"]["local"],args["dift_code_config"]["local_noalbedo"][0])
        self.g_diff_global_part = DIFT_NET_G_DIFF_GLOBAL(args,args["partition"]["global"],args["dift_code_config"]["global"][0])
        # self.g_spec_part = DIFT_NET_G_SPEC(args,args["partition"]["g_spec"][0],args["partition"]["g_spec"][1])
        
    def forward(self,batch_data,view_mat_model_t,view_mat_for_normal_t,return_origin_codes=False):
        '''
        batch_data=(batch_size,sample_view_num,m_len,1)
        view_mat_model_t = (batch_size,16)
        view_mat_for_normal_t = (batch_size,16)
        albedo_diff = (batch_size,1)
        albedo_spec = (batch_size,1)
        '''
        batch_size = batch_data.size()[0]
        device = batch_data.device

        x_n = batch_data.reshape(batch_size,self.measurements_length)
        
        m_ptr = 0
    
        dift_codes_albedo = self.albedo_part(x_n[:,m_ptr:m_ptr+self.partition["local"]])

        dift_codes_g_diff_local = self.g_diff_local_part(x_n[:,m_ptr:m_ptr+self.partition["local"]])
        m_ptr += self.partition["local"]

        dift_codes_g_diff_global = self.g_diff_global_part(x_n[:,m_ptr:m_ptr+self.partition["global"]],view_mat_model_t,view_mat_for_normal_t)
        m_ptr+=self.partition["global"]

        # dift_codes_g_spec = self.g_spec_part(m_no_rhos[:,m_ptr:m_ptr+self.partition["g_spec"][0]],view_mat_model_t,view_mat_for_normal_t)
        # m_ptr+=self.partition["g_spec"][0]


        dift_codes = torch.cat([dift_codes_albedo,dift_codes_g_diff_local,dift_codes_g_diff_global],dim=1)

        if not return_origin_codes:
            return dift_codes#,torch.zeros(batch_size,3,dtype=torch.float32)
        else:
            origin_codes_map = {
                "local_albedo" : dift_codes_albedo,
                "local_noalbedo" : dift_codes_g_diff_local,
                "global" : dift_codes_g_diff_global,
                # "g_spec" : dift_codes_g_spec,
            }#CAUTION THIS MUST BE SAME AS DIFT_CODE_CONFIG
            return dift_codes,origin_codes_map