import numpy as np
import cv2
import os
import argparse

def read_imshow(args,view_id):
    cur_img = cv2.imread(args.data_root+"pd_predicted_{}_0.png".format(view_id))
    cv2.imshow("img",cur_img)

    return cur_img

cur_img = None
cur_x = None
cur_y = None

def draw_cross_on_img(img,x,y,bold = 5):
    # cv2.line(cur_img,(0,0),(x,y),(255,255,0),2)
    img[y,x-bold//2:x+bold//2+1] = 255.0
    img[y-bold//2:y+bold//2+1,x] = 255.0
    return img

def OnMouseAction(event,x,y,flags,param):
    global cur_img,cur_x,cur_y

    if event == cv2.EVENT_LBUTTONDOWN:
        cur_img_crossed = draw_cross_on_img(cur_img.copy(),x,y)
        cur_x = x
        cur_y = y
        print("cur_x:{},cur_y:{}".format(cur_x,cur_y))
        cv2.imshow("img",cur_img_crossed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,default="E:/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/pot2PNG/full_on_udt/")
    parser.add_argument("--view_num",type=int,default=20)

    args = parser.parse_args()

    save_root = args.data_root+"marked_points/"
    os.makedirs(save_root,exist_ok=True)

    cur_view_id = 0
    cur_img = read_imshow(args,cur_view_id)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img',OnMouseAction)

    cc_list = []
    point_counter = 0

    should_update_img = False
    should_add_point = False
    while True:
        key = cv2.waitKey()
        origin_view_id = cur_view_id
        if key == ord('q'):
            break
        elif key == ord('d'):
            cur_view_id = (cur_view_id + 1) % args.view_num
            should_update_img = True
        elif key == ord('a'):
            cur_view_id = (cur_view_id - 1 + args.view_num) % args.view_num
            should_update_img = True
        elif key == ord('m'):
            should_add_point = True
        elif key == ord('o'):
            with open(save_root+"{}.txt".format(point_counter),"w") as pf:
                for view,tmpx,tmp_y in cc_list:
                    pf.write("{},{},{}\n".format(view,tmpx,tmp_y))
            cc_list = []
            point_counter+=1
        else:
            pass
        
        if should_add_point:
            cc_list.append((origin_view_id,cur_x,cur_y))
            print("cc_list:",cc_list)
            should_add_point = False


        if should_update_img:
            cur_img = read_imshow(args,cur_view_id)
            should_update_img = False
            print("cur_view_id:{}".format(cur_view_id))