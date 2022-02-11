import torch
import numpy as np
import os
import argparse
import time
import torch

def lighting_pattern_process(kernel):
    # kernel = torch.sigmoid(kernel)#[measurementnum,lightnum]
    kernel = torch.nn.functional.normalize(kernel,dim=1)
    return kernel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")
    parser.add_argument("node_name")
    parser.add_argument("sample_view_num",type=int)

    args = parser.parse_args()

    checkpoint = torch.load(args.model_root + args.model_file_name, map_location='cpu')
    
    key_collector = []
    for a_key in checkpoint:
        print(a_key)
        if args.node_name in a_key:
            for which_view in range(args.sample_view_num):
                key_collector.append(a_key)

    assert len(key_collector) == args.sample_view_num,"expect {} linear projection kernel,got {}".format(args.sample_view_num,len(key_collector))

    print("[WARNING] we load the unadulterated parameters from model, you may processed that in forward func.")
    print("[WARNING] You need to process them here too.")
    print("[WARNING] Understood?(y/n)")
    if input() != "y":
        exit(-1)

    for which_view,a_key in enumerate(key_collector):
        cur_save_root = args.model_root+"{}/".format(which_view)
        os.makedirs(cur_save_root,exist_ok=True)
        lighting_pattern = checkpoint[a_key]

        lighting_pattern = lighting_pattern_process(lighting_pattern)
        lighting_pattern = lighting_pattern.numpy()

        print("origin lighting pattern param shape:",lighting_pattern.shape)

        if len(lighting_pattern.shape) == 2:
            lighting_pattern = np.expand_dims(lighting_pattern,axis=-1)
        #(pattern_num,light_num,channel_num)
        if lighting_pattern.shape[-1] == 1:
            lighting_pattern = np.repeat(lighting_pattern,3,axis=-1)
        #(pattern_num,light_num,3)

        print("W.bin shape:",lighting_pattern.shape)
        lighting_pattern.astype(np.float32).tofile(cur_save_root+"W.bin")