import torch
import numpy as np
import argparse
import os
import math
import scipy.io as scio
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
    parser.add_argument("dift_code_len_g",type=int)
    parser.add_argument("dift_code_len_m",type=int)

    parser.add_argument("--batch_size",type=int,default=5000)
    parser.add_argument("--scalar",type=float,default=1e-5)

    args = parser.parse_args()

    inference_device = torch.device("cuda:3")

    ################################################
    #####load net
    #################################################
    # #about rendering devices

    nn_model = DIFT_NET_inuse(args)
    pretrained_dict = torch.load(args.model_root + args.model_file_name, map_location=inference_device)
    print("loading trained model...")
    model_dict = nn_model.state_dict()
    something_not_found=False
    for k,_ in model_dict.items():
        if k not in pretrained_dict:
            print("not found:",k)
            something_not_found = True
    if something_not_found:
        exit()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    nn_model.load_state_dict(model_dict)
    nn_model.to(inference_device)
    nn_model.eval()
    
    ################################################
    ####prepare for saving
    ################################################
    save_root = args.root+"feature_maps/images_0/"
    os.makedirs(save_root,exist_ok=True)
    ################################################
    ####infer here
    ################################################
    sampled_rotate_angles_np = np.linspace(0.0,-math.pi*2.0,num=args.sample_view_num,endpoint=False)
    sampled_rotate_angles_np = np.expand_dims(sampled_rotate_angles_np,axis=0).astype(np.float32)

    calib_mat = scio.loadmat(args.root+"Calib_Results.mat")

    for which_view in range(args.rotate_num):
        print("view{}/{}".format(which_view,args.rotate_num-1))
        cur_root = args.root+"view_{:02d}/".format(which_view+1)
        cur_save_root = save_root+"{}/".format(which_view)
        os.makedirs(cur_save_root,exist_ok=True)
        measurments = np.fromfile(cur_root+"cam00_data_96_nocc_compacted.bin".format(args.measurement_len),np.float32).reshape([-1,3,96])
        measurments = measurments.reshape(-1,96)[:,::96//args.measurement_len]
        measurments = measurments * args.scalar

        R_matrix = calib_mat["Rc_{}".format(which_view+1)].astype(np.float32).reshape((1,-1))
        T_vec = calib_mat["Tc_{}".format(which_view+1)].astype(np.float32).reshape((1,-1))
        rt_vec = np.concatenate((R_matrix, T_vec),axis=1)
        # print(rt_vec)

        pf_save = open(cur_save_root+"feature.bin".format(which_view),"wb")
        pf_save_normal = open(cur_save_root+"normal.bin".format(which_view),"wb")

        ptr = 0
        while True:
            tmp_measurements = measurments[ptr:ptr+args.batch_size]
            cur_batch_size = tmp_measurements.shape[0]
            if cur_batch_size == 0:
                break
            tmp_measurements = torch.from_numpy(tmp_measurements).to(inference_device)
            sampled_rotate_angles = torch.from_numpy(rt_vec).repeat(cur_batch_size,1).to(inference_device)
            with torch.no_grad():
                dift_codes = nn_model(tmp_measurements,sampled_rotate_angles)
            
            dift_codes = dift_codes.cpu().numpy()
            dift_codes.astype(np.float32).tofile(pf_save)

            # normal_nn = normal_nn.cpu().numpy()
            # normal_nn.astype(np.float32).tofile(pf_save_normal)

            ptr+=cur_batch_size
        
        pf_save.close()
        pf_save_normal.close()
        
