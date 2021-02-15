import numpy as np
import cv2
import os

data_root = "/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/readingPNG/"
view_num = 20
light_num = 12*8

save_root = data_root + "full_on_udt/"
os.makedirs(save_root,exist_ok=True)

for which_view in range(view_num):
    cur_root = data_root + "view_{:02d}/".format(which_view+1)
    origin_img = cv2.imread(cur_root+"{:03d}.png".format(1))
    origin_img = origin_img.astype(np.float32)
    for which_light in range(1,light_num):
        img = cv2.imread(cur_root+"{:03d}.png".format(which_light+1))
        img = img.astype(np.float32)
        origin_img = origin_img+img
    
    origin_img = origin_img*0.04
    origin_img = np.clip(origin_img,0.0,255.0)
    origin_img = origin_img.astype(np.uint8)

    cv2.imwrite(save_root+"pd_predicted_{}_0.png".format(which_view),origin_img)