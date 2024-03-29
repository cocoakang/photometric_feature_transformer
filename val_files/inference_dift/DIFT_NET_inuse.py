import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import argparse
import sys
sys.path.append("../../")
TORCH_RENDER_PATH="../../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
from DIFT_NET import DIFT_NET

class DIFT_NET_inuse(nn.Module):
    def __init__(self,args,setup):
        super(DIFT_NET_inuse,self).__init__()
        ########################################
        ##parse configuration                ###
        ########################################
        
        self.measurement_len = args.measurement_len
        self.setup = setup

        partition = {
            "local":5,
            "global":3
        }

        dift_code_config = {
            "local_noalbedo":(9,-1.0),
            "global":(7,-1.0),
            "cat":(16,-1.0)
        }

        training_configs = {
            "partition": partition,
            "dift_code_config" : dift_code_config,
            "measurements_length" : args.measurement_len,
            "dift_code_len" : 10,
            "training_mode" : "finetune"
        }

        ########################################
        ##construct net                      ###
        ########################################
        self.dift_net = DIFT_NET(training_configs)
        # self.dift_net_m = DIFT_NET_M(training_configs)

    def forward(self,batch_data,sampled_rotate_angles):
        '''
        batch_data = (batch_size,m_len,1)
        '''
        batch_size = batch_data.size()[0]

        cossin = torch.cat(
                [
                    torch.sin(sampled_rotate_angles),
                    torch.cos(sampled_rotate_angles)
                ],dim=1
            )
        
        # view_mat_model = torch_render.rotation_axis(-sampled_rotate_angles,self.setup.get_rot_axis_torch(batch_data.device))#[2*batch,4,4]
        # view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]
        # view_mat_model_t = view_mat_model_t.reshape(batch_size,16)
        # view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
        # view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[2*batch,4,4]
        # view_mat_for_normal_t = view_mat_for_normal_t.reshape(batch_size,16)

        # infered_dift_codes_g = self.dift_net(batch_data,cossin,view_mat_model_t,view_mat_for_normal_t)#(batch,3)
        # infered_dift_codes_m = self.dift_net_m(batch_data,cossin,view_mat_model_t,view_mat_for_normal_t)#(batch,3)
        # infered_dift_codes = torch.cat([infered_dift_codes_g,infered_dift_codes_m],dim=1)

        infered_dift_codes = self.dift_net(batch_data,cossin)

        return infered_dift_codes