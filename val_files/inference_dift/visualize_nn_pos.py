import torch
import numpy as np
import argparse
import cv2
import os
from DIFT_NET_inuse import DIFT_NET_inuse
from sklearn.decomposition import PCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")

    parser.add_argument("rotate_num",type=int)
    parser.add_argument("dift_code_len",type=int,choices=[3])

    parser.add_argument("--batch_size",type=int,default=5000)
    parser.add_argument("--imgheight",type=int,default=3000)
    parser.add_argument("--imgwidth",type=int,default=4096)

    args = parser.parse_args()

    log_folder = args.root+"nn_positions/"
    #######step 1 read in all features#########
    point_num = 0
    for which_view in range(args.rotate_num):
        with open(log_folder+"{}/pos.bin".format(which_view)) as pf:
            pf.seek(0,2)
            tmp_point_num = pf.tell()//4//args.dift_code_len
            point_num+= tmp_point_num
            print("view:{} pointnum:{}".format(which_view,tmp_point_num))
    print("total point num:",point_num)       

    features = np.zeros([point_num,args.dift_code_len],np.float32)  
    ptr = 0
    for which_view in range(args.rotate_num):
        print("loading data view:{}".format(which_view))
        tmp_features = np.fromfile(log_folder+"{}/pos.bin".format(which_view),np.float32).reshape([-1,args.dift_code_len])
        # tmp_features = tmp_features[np.random.randint(tmp_features.shape[0], size=tmp_features.shape[0]//2), :].copy()
        features[ptr:ptr+tmp_features.shape[0]] = tmp_features
        ptr+=tmp_features.shape[0]

    ######step 3 draw features#################
    feature_img_folder = args.root+"nn_positions/"
    ptr = 0
    # feature_file_bin_head = np.array([args.imgwidth,args.imgheight,test_dim],np.int32)
    for which_view in range(args.rotate_num):
        with open(args.root+"{}/cam00_index_nocc.bin".format(which_view),"rb") as pf:
            pixel_num = np.fromfile(pf,np.int32,1)[0]
            idxes = np.fromfile(pf,np.int32).reshape([-1,2])
            assert idxes.shape[0] == pixel_num
            print("pixel num:{}".format(pixel_num))

        feature_origin = np.fromfile(log_folder+"{}/pos.bin".format(which_view),np.float32).reshape([-1,args.dift_code_len])

        #######visualize feature images
        tmp_img = np.zeros([args.imgheight,args.imgwidth,3],np.float32)
        tmp_img[idxes[:,1],idxes[:,0]] = feature_origin

        cv2.imwrite(feature_img_folder+"pos_{}.exr".format(which_view),tmp_img)
    
        ######save feature bin for colmap
        # tmp_img = np.zeros([args.imgheight,args.imgwidth,test_dim],np.float32)
        # tmp_features = pca2.transform(feature_origin)
        # tmp_img[idxes[:,1],idxes[:,0]] = tmp_features
        
        # with open(current_folder+"pd_predicted_{}_{}.png.bin".format(which_view,which_cam),"wb") as p_feature_file_bin:
        #     feature_file_bin_head.tofile(p_feature_file_bin)
        #     tmp_img.astype(np.float32).tofile(p_feature_file_bin)
