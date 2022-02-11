import numpy as np
import cv2
import os
import sys
TORCH_RENDER_PATH = "../../../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render

lumitexel_size = 24576

def quantize_pattern(pattern_float):
    if pattern_float.dtype != np.float32:
        print("[QUATIZE ERROR]error input type:",pattern_float.dtype)
    pattern_quantized = pattern_float*255.0
    pattern_quantized = pattern_quantized.astype(np.uint8)
    return pattern_quantized

if __name__ == "__main__":
    modelPath = sys.argv[1]
    sample_view_num = int(sys.argv[2])
    img_root = modelPath+"imgs/"
    os.makedirs(img_root,exist_ok=True)

    for which_view in range(sample_view_num):
        cur_save_root = modelPath+"{}/".format(which_view)
        raw_pattern = np.fromfile(cur_save_root+"W_flipped.bin",np.float32).reshape([-1,lumitexel_size,3])
        pattern_num = raw_pattern.shape[0]

        maxs = np.zeros([pattern_num],np.float32)
        quantized_pattern = np.zeros(raw_pattern.shape,np.uint8)
        for idx,a_pattern in enumerate(raw_pattern):
            # print(a_pattern.max())
            maxs[idx] = a_pattern.max()
            quantized_pattern[idx] = quantize_pattern(a_pattern/maxs[idx])
        maxs.tofile(cur_save_root+"maxs.bin")
        np.savetxt(cur_save_root+"maxs.txt",maxs,delimiter=',',fmt='%.3f')
        quantized_pattern.tofile(cur_save_root+"W_quantized.bin")

        cur_img_root = img_root+"{}/".format(which_view)
        os.makedirs(cur_img_root,exist_ok=True)

        standard_rendering_parameters = {}

        standard_rendering_parameters = {
                "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
            }
        setup = torch_render.Setup_Config(standard_rendering_parameters)
        
        with open(cur_save_root+"W_quantized.bin","rb") as f:
            data = np.fromfile(f,np.uint8).reshape([-1,lumitexel_size,3])
            img = torch_render.visualize_lumi(data,setup)
            for i in range(img.shape[0]):
                cv2.imwrite(cur_img_root+"W_{}.png".format(i),img[i][:,:,::-1])