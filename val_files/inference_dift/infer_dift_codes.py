import torch
import numpy as np
import argparse
import os
from DIFT_NET_inuse import DIFT_NET_inuse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("model_root")
    parser.add_argument("model_file_name")

    parser.add_argument("rotate_num",type=int)
    parser.add_argument("measurement_len",type=int)
    parser.add_argument("dift_code_len",type=int)

    parser.add_argument("--batch_size",type=int,default=5000)
    parser.add_argument("--scalar",type=float,default=184.0)

    args = parser.parse_args()

    ################################################
    #####load net
    ################################################
    nn_model = DIFT_NET_inuse(args)
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
    for which_view in range(args.rotate_num):
        print("view{}/{}".format(which_view,args.rotate_num-1))
        cur_root = args.root+"{}/".format(which_view)
        cur_save_root = save_root+"{}/".format(which_view)
        os.makedirs(cur_save_root,exist_ok=True)
        measurments = np.fromfile(cur_root+"cam00_data_{}_nocc_compacted.bin".format(args.measurement_len*2),np.float32).reshape([-1,3,args.measurement_len])
        measurments = measurments * args.scalar
        measurments = np.mean(measurments,axis=1)

        pf_save = open(cur_save_root+"feature.bin".format(which_view),"wb")

        ptr = 0
        while True:
            tmp_measurements = measurments[ptr:ptr+args.batch_size]
            cur_batch_size = tmp_measurements.shape[0]
            if cur_batch_size == 0:
                break
            tmp_measurements = torch.from_numpy(tmp_measurements).to("cuda:0")
            with torch.no_grad():
                dift_codes = nn_model(tmp_measurements)
            
            dift_codes = dift_codes.cpu().numpy()
            dift_codes.astype(np.float32).tofile(pf_save)

            ptr+=cur_batch_size
        
        pf_save.close()
        
