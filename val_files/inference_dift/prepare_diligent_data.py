import numpy as np
import cv2
import argparse
import os

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/readingPNG/")
    parser.add_argument("--lightnum",type=int,default=12*8)
    parser.add_argument("--view_num",type=int,default=20)

    args = parser.parse_args()

    for which_view in range(args.view_num):
        print(which_view)
        cur_data_root = args.data_root+"view_{:02d}/".format(which_view+1)

        mask = cv2.imread(cur_data_root+"mask.png")[:,:,0]
        valid_idxes = np.stack(np.where(mask > 0),axis=1)[:,::-1]#(pointnum,2) x,y

        measurement_collector = []
        for which_light in range(args.lightnum):
            tmp_img = cv2.imread(cur_data_root+"{:03d}.png".format(which_light+1))[:,:,::-1]
            tmp_measurements = tmp_img[valid_idxes[:,1],valid_idxes[:,0]]
            measurement_collector.append(tmp_measurements)
        
        measurement_collector = np.stack(measurement_collector,axis=2)#(pointnum,3,lightnum)
        measurement_collector = measurement_collector.astype(np.float32).tofile(cur_data_root+"cam00_data_{}_nocc_compacted.bin".format(args.lightnum))

        valid_idxes.astype(np.int32).tofile(cur_data_root+"cam00_index_nocc.bin")

        print("done.")