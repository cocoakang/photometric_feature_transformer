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
from DIFT_NET_NORMAL_inuse import DIFT_NET_NORMAL_inuse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/buddhaPNG//view_01/")
    parser.add_argument("--model_root",default="/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/dift_extractor/runs/diligent_normal/models/")
    parser.add_argument("--model_file_name",default="model_state_60000.pkl")

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
    nn_model = DIFT_NET_NORMAL_inuse(args)
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
    nn_model.to("cuda:1")
    nn_model.eval()
      
    measurments = np.fromfile(args.data_root+"measurements.bin",np.float32).reshape([-1,args.measurement_len,1])

    pf_save = open(args.data_root+"normal_infered.bin","wb")

    ptr = 0
    while True:
        tmp_measurements = measurments[ptr:ptr+args.batch_size]
        cur_batch_size = tmp_measurements.shape[0]
        if cur_batch_size == 0:
            break
        tmp_measurements = torch.from_numpy(tmp_measurements).to("cuda:1")
        with torch.no_grad():
            dift_codes = nn_model(tmp_measurements)
        
        dift_codes = dift_codes.cpu().numpy()
        dift_codes.astype(np.float32).tofile(pf_save)

        ptr+=cur_batch_size
    
    pf_save.close()

    idxes = np.fromfile(args.data_root+"indexes.bin",np.int32).reshape((-1,2))

    normals = np.fromfile(args.data_root+"normal_infered.bin",np.float32).reshape((-1,3))        

    img = np.zeros((512,612,3),np.float32)

    img[idxes[:,0],idxes[:,1]] = normals*0.5+0.5

    cv2.imwrite(args.data_root+"normal_infered.png",img[:,:,::-1]*255.0)