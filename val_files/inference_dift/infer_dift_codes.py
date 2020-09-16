import torch
import numpy as np
import argparse
import os
import math
import sys
TORCH_RENDER_PATH="../../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import Setup_Config
from DIFT_NET_inuse import DIFT_NET_inuse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")

    parser.add_argument("sample_view_num",type=int)
    parser.add_argument("rotate_num",type=int)
    parser.add_argument("measurement_len",type=int)
    parser.add_argument("dift_code_len",type=int)
    parser.add_argument("view_code_len",type=int)

    parser.add_argument("--batch_size",type=int,default=5000)
    parser.add_argument("--scalar",type=float,default=30.0)#184.0)

    args = parser.parse_args()

    ################################################
    #####load net
    #################################################
    # #about rendering devices
    standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
    setup_input = Setup_Config(standard_rendering_parameters)

    nn_model = DIFT_NET_inuse(args,setup_input)
    pretrained_dict = torch.load(args.model_root + args.model_file_name, map_location='cuda:0')
    print("loading trained model...")
    model_dict = nn_model.state_dict()
    something_not_found=False
    for k,_ in model_dict.items():
        if k not in pretrained_dict and "linear_projection" not in k:
            print("not found:",k)
            something_not_found = True
    if something_not_found:
        exit()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    nn_model.load_state_dict(model_dict)
    nn_model.to("cuda:0")
    nn_model.eval()

    ################################################
    ####prepare for saving
    ################################################
    save_root = args.root+"feature_maps/images_0/"
    os.makedirs(save_root,exist_ok=True)
    ################################################
    ####infer here
    ################################################
    sampled_rotate_angles_np = np.linspace(0.0,math.pi*2.0,num=args.sample_view_num,endpoint=False)
    sampled_rotate_angles_np = np.expand_dims(sampled_rotate_angles_np,axis=0).astype(np.float32)

    for which_view in range(args.rotate_num):
        print("view{}/{}".format(which_view,args.rotate_num-1))
        cur_root = args.root+"{}/".format(which_view)
        cur_save_root = save_root+"{}/".format(which_view)
        os.makedirs(cur_save_root,exist_ok=True)
        measurments = np.fromfile(cur_root+"cam00_data_{}_nocc_compacted.bin".format(args.measurement_len*2),np.float32).reshape([-1,args.measurement_len,3])
        measurments = np.transpose(measurments,(0,2,1)).reshape(-1,args.measurement_len)
        measurments = measurments * args.scalar

        pf_save = open(cur_save_root+"feature.bin".format(which_view),"wb")
        pf_save_normal = open(cur_save_root+"normal.bin".format(which_view),"wb")

        ptr = 0
        while True:
            tmp_measurements = measurments[ptr:ptr+args.batch_size]
            cur_batch_size = tmp_measurements.shape[0]
            if cur_batch_size == 0:
                break
            tmp_measurements = torch.from_numpy(tmp_measurements).to("cuda:0")
            sampled_rotate_angles = torch.from_numpy(sampled_rotate_angles_np).repeat(cur_batch_size,1).to("cuda:0")
            sampled_rotate_angles = sampled_rotate_angles[:,[which_view]]
            with torch.no_grad():
                dift_codes = nn_model(tmp_measurements,sampled_rotate_angles)
            
            dift_codes = dift_codes.cpu().numpy()
            dift_codes.astype(np.float32).tofile(pf_save)

            # normal_nn = normal_nn.cpu().numpy()
            # normal_nn.astype(np.float32).tofile(pf_save_normal)

            ptr+=cur_batch_size
        
        pf_save.close()
        pf_save_normal.close()
        
