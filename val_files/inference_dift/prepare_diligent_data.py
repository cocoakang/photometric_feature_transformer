import numpy as np
import cv2
import argparse
import os

if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/buddhaPNG//view_01/")
    parser.add_argument("--lightnum",type=int,default=12*8)

    args = parser.parse_args()

    mask = cv2.imread(args.data_root+"mask.png")[:,:,0]
    valid_idxes = np.stack(np.where(mask > 0),axis=1)#(pointnum,2) y,x

    measurement_collector = []
    for which_light in range(args.lightnum):
        tmp_img = cv2.imread(args.data_root+"{:03d}.png".format(which_light+1),cv2.IMREAD_GRAYSCALE)
        tmp_measurements = tmp_img[valid_idxes[:,0],valid_idxes[:,1]]
        measurement_collector.append(tmp_measurements)
    
    measurement_collector = np.stack(measurement_collector,axis=1)#(pointnum,lightnum)
    measurement_collector = measurement_collector.astype(np.float32).tofile(args.data_root+"measurements.bin")

    valid_idxes.astype(np.int32).tofile(args.data_root+"indexes.bin")

    print("done.")