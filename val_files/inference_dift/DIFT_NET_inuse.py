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
        # batch_size = batch_data.size()[0]

        cossin = torch.cat(
                [
                    torch.sin(sampled_rotate_angles),
                    torch.cos(sampled_rotate_angles)
                ],dim=1
            )

        infered_dift_codes = self.dift_net(batch_data,cossin)#(batch,3)
        dift_codes_origin = torch_render.rotate_vector_along_axis(self.setup,-sampled_rotate_angles,infered_dift_codes)

        return dift_codes_origin