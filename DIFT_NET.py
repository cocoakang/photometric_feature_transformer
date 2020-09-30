import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import math
from DIFT_NET_ALBEDO import DIFT_NET_ALBEDO
# from DIFT_NET_G_DIFF_LOCAL import DIFT_NET_G_DIFF_LOCAL
# from DIFT_NET_G_DIFF_GLOBAL import DIFT_NET_G_DIFF_GLOBAL
# from DIFT_NET_G_SPEC import DIFT_NET_G_SPEC

class DIFT_NET(nn.Module):
    def __init__(self,args):
        super(DIFT_NET,self).__init__()
    
        self.measurements_length = args["measurements_length"]
        self.dift_code_len = args["dift_code_len"]
        self.keep_prob = 0.9
        #############construct model
        input_size = self.measurements_length*1#+self.view_code_len
        
        self.albedo_part = DIFT_NET_ALBEDO(args,2,args["partition"]["albedo"][1])
        # self.g_diff_local_part = DIFT_NET_G_DIFF_LOCAL(args,args["patition"]["g_diff_local"][0],args["patition"]["g_diff_local"][1])
        # self.g_diff_global_part = DIFT_NET_G_DIFF_GLOBAL(args,args["patition"]["g_diff_global"][0],args["patition"]["g_diff_global"][1])
        # self.g_spec_part = DIFT_NET_G_SPEC(args,args["patition"]["g_spec"][0],args["patition"]["g_spec"][1])
        
    def forward(self,batch_data,view_mat_model_t,view_mat_for_normal_t,albedo_diff,albedo_spec,return_origin_codes=False):
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
        
        m_no_rhod = x_n / (1e-6 + albedo_diff)
        m_no_rhos = x_n / (1e-6 + albedo_spec)
        
        albedo = torch.cat((albedo_diff,albedo_spec),dim=1)#(batchsize,2)

        dift_codes_albedo = self.albedo_part(albedo)

        # dift_codes = self.dift_part(x_n)
        # dift_codes = torch.nn.functional.normalize(dift_codes,dim=1)

        # dift_codes = self.dift_part2(dift_codes)
        # dift_codes = torch.nn.functional.normalize(dift_codes,dim=1)

        dift_codes = dift_codes_albedo
        dift_codes = torch.nn.functional.normalize(dift_codes,dim=1)

        if not return_origin_codes:
            return dift_codes#,torch.zeros(batch_size,3,dtype=torch.float32)
        else:
            origin_codes_map = {
                "albedo":dift_codes_albedo
            }
            return dift_codes,origin_codes_map