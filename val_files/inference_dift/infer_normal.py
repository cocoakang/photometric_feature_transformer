import torch
import numpy as np
import argparse
import os
import cv2
import math
import sys
TORCH_RENDER_PATH="../../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
from torch_render import Setup_Config
from DIFT_NET_inuse import DIFT_NET_inuse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/buddhaPNG/")
    parser.add_argument("--model_root",default="/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/log_no_where2/models/")
    parser.add_argument("--model_file_name",default="model_state_30000.pkl")
    parser.add_argument("--view_num",type=int,default=20)

    # parser.add_argument("sample_view_num",type=int)
    # parser.add_argument("rotate_num",type=int)
    parser.add_argument("--measurement_len",type=int,default=12*8)
    # parser.add_argument("dift_code_len",type=int,choices=[3])
    # parser.add_argument("view_code_len",type=int)

    parser.add_argument("--batch_size",type=int,default=5000)

    args = parser.parse_args()

    ################################################
    #####load net
    #################################################
    nn_model = DIFT_NET_inuse(args)
    pretrained_dict = torch.load(args.model_root + args.model_file_name, map_location='cuda:3')
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
    nn_model.to("cuda:3")
    nn_model.eval()

    save_root = args.data_root+"infered_normals/"
    os.makedirs(save_root,exist_ok=True)

    for which_view in range(args.view_num):
        cur_root = args.data_root+"view_{:02d}/".format(which_view+1)

        measurments = np.fromfile(cur_root+"cam00_data_{}_nocc_compacted.bin".format(args.measurement_len),np.float32).reshape([-1,3,args.measurement_len])
        measurments = np.mean(measurments,axis=1)

        pf_save = open(cur_root+"normal_infered.bin","wb")

        ptr = 0
        while True:
            tmp_measurements = measurments[ptr:ptr+args.batch_size]
            cur_batch_size = tmp_measurements.shape[0]
            if cur_batch_size == 0:
                break
            tmp_measurements = torch.from_numpy(tmp_measurements).to("cuda:3")
            with torch.no_grad():
                dift_codes = nn_model(tmp_measurements,None,get_normal=True)
            
            dift_codes = dift_codes.cpu().numpy()
            dift_codes.astype(np.float32).tofile(pf_save)

            ptr+=cur_batch_size
        
        pf_save.close()

        idxes = np.fromfile(cur_root+"cam00_index_nocc.bin",np.int32).reshape((-1,2))

        normals = np.fromfile(cur_root+"normal_infered.bin",np.float32).reshape((-1,3))        

        img = np.zeros((512,612,3),np.float32)

        img[idxes[:,0],idxes[:,1]] = normals*0.5+0.5

        cv2.imwrite(save_root+"normal_infered_{}.png".format(which_view),img[:,:,::-1]*255.0)