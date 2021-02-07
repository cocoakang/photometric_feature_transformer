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
from DIFT_NET_NORMAL import DIFT_NET_NORMAL

class DIFT_NET_NORMAL_inuse(nn.Module):
    def __init__(self,args):
        super(DIFT_NET_NORMAL_inuse,self).__init__()
        ########################################
        ##parse configuration                ###
        ########################################
        
        self.measurement_len = args.measurement_len

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
        self.dift_net = DIFT_NET_NORMAL(training_configs)
        # self.dift_net_m = DIFT_NET_M(training_configs)

    def forward(self,batch_data):
        '''
        batch_data = (batch_size,m_len,1)
        '''
        batch_size = batch_data.size()[0]

        infered_normal = self.dift_net(batch_data)

        return infered_normal