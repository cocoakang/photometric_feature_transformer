import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import argparse
import sys
sys.path.append("../../")
from DIFT_NET import DIFT_NET

class DIFT_NET_inuse(nn.Module):
    def __init__(self,args):
        super(DIFT_NET_inuse,self).__init__()
        ########################################
        ##parse configuration                ###
        ########################################
        
        self.measurement_len = args.measurement_len
        self.dift_code_len = args.dift_code_len

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

        infered_dift_codes = self.dift_net(batch_data,sampled_rotate_angles)#(batch,3)

        return infered_dift_codes