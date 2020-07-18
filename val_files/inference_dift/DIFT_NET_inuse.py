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
        self.dift_code_len = args.dift_code_len
        self.setup = setup

        training_configs = {
            "measurements_length" : args.measurement_len,
            "dift_code_len" : args.dift_code_len,
            "view_code_len" : args.view_code_len
        }

        ########################################
        ##construct net                      ###
        ########################################
        self.dift_net = DIFT_NET(training_configs)

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
        
        view_mat_model = torch_render.rotation_axis(-sampled_rotate_angles,self.setup.get_rot_axis_torch(batch_data.device))#[2*batch,4,4]
        view_mat_model_t = torch.transpose(view_mat_model,1,2)#[batch,4,4]
        view_mat_model_t = view_mat_model_t.reshape(batch_size,16)
        view_mat_for_normal =torch.transpose(torch.inverse(view_mat_model),1,2)
        view_mat_for_normal_t = torch.transpose(view_mat_for_normal,1,2)#[2*batch,4,4]
        view_mat_for_normal_t = view_mat_for_normal_t.reshape(batch_size,16)

        infered_dift_codes,normal_global_nn = self.dift_net(batch_data,cossin,view_mat_model_t,view_mat_for_normal_t)#(batch,3)
        # dift_codes_origin = torch_render.rotate_vector_along_axis(self.setup,-sampled_rotate_angles,infered_dift_codes)
        # infered_dift_codes = torch_render.rotate_point_along_axis(self.setup,-sampled_rotate_angles,infered_dift_codes)

        return infered_dift_codes,normal_global_nn