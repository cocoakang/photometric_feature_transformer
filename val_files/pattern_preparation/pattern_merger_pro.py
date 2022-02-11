import numpy as np
import cv2
import os
import argparse
import shutil

def gcd(a,b):
    a,b = max(a,b),min(a,b)
    a,b = b,a%b
    while b:
        a, b = b, a%b
    return a

def lcm(a,b):
	return a*b//gcd(a,b)

def extract_model_name(full_path):
    
    end = full_path.find("/models/")
    start = full_path[:end].rfind("/")
    return full_path[start+1:end]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("save_root")
    parser.add_argument("model_num",type=int)
    parser.add_argument('model_paths', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    os.makedirs(args.save_root,exist_ok=True)
    save_root = args.save_root+"shot_patterns/"
    os.makedirs(save_root,exist_ok=True)
    img_root = save_root+"imgs/"
    os.makedirs(img_root,exist_ok=True)

    sample_view_num_list = []

    pf_readme = open(args.save_root+"readme.txt","w")
    pf_readme.write("拍摄一个任务，{}模型，\n".format(args.model_num))
    for which_model in range(args.model_num):
        tmp_path = args.model_paths[which_model]
        model_name = extract_model_name(tmp_path)
        pf_readme.write("{} 模型路径: {}\n".format(model_name,tmp_path))
        sample_view_num = len(os.listdir(args.model_paths[which_model]+"imgs/"))
        print("model:",args.model_paths[which_model],"sample view num:",sample_view_num)
        sample_view_num_list.append(sample_view_num)
    pf_readme.close()

    cur = sample_view_num_list[0]
    for which_model in range(1,args.model_num):
        cur = lcm(cur,sample_view_num_list[which_model])
    
    rotate_num = cur

    for which_view in range(rotate_num):
        cur_img_root = img_root+"{}/".format(which_view)
        os.makedirs(cur_img_root,exist_ok=True)
        pf_num = open(cur_img_root+"num.txt","w")
        total_counter = 0
        for which_model in range(args.model_num):
            ratio = rotate_num//sample_view_num_list[which_model]
            if which_view % ratio != 0:
                continue
            sample_view_id = which_view//ratio
            tmp_path = args.model_paths[which_model]
            model_name = extract_model_name(tmp_path)
            
            img_from_root = tmp_path+"imgs/{}/".format(sample_view_id)
            img_num = len(os.listdir(img_from_root))
            for which_img in range(img_num):
                shutil.copyfile(img_from_root+"W_{}.png".format(which_img),cur_img_root+"W_{}.png".format(total_counter))
                total_counter+=1
            pf_num.write("{} {}\n".format(model_name,img_num))
        pf_num.close()