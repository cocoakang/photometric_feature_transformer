import numpy as np
TORCH_RENDER_PATH="../torch_renderer/"
import sys
sys.path.append(TORCH_RENDER_PATH)
import torch_render
import torch
import os
import cv2
from torch_render import Setup_Config_Freeform

if __name__ == "__main__":
    config = {
        "config_dir":TORCH_RENDER_PATH + "wallet_of_torch_renderer/diligent_mv2/"
    }
    setup = Setup_Config_Freeform(config)

    sample_num = 10

    test_positions = torch.cat(
        [
            torch.ones(sample_num,1,dtype=torch.float64)*0.0,#n2
            torch.ones(sample_num,1,dtype=torch.float64)*0.0,#t
            torch.ones(sample_num,1,dtype=torch.float64)*0.0,#ax
        ],dim=1
    )
    test_params = torch.cat(
        [
            torch.ones(sample_num,2,dtype=torch.float64)*0.5,#n2
            torch.ones(sample_num,1,dtype=torch.float64)*0.5,#t
            torch.ones(sample_num,1,dtype=torch.float64)*0.1,#ax
            torch.ones(sample_num,1,dtype=torch.float64)*0.006,#ay
            torch.ones(sample_num,1,dtype=torch.float64)*0.5,#rhod
            torch.ones(sample_num,1,dtype=torch.float64)*0.0,#rhos
        ],dim=1
    )
    test_rotation = torch.zeros(sample_num,1,dtype=torch.float64)

    rendered_lumi,_ = torch_render.draw_rendering_net(setup,test_params,test_positions,test_rotation,"test")

    rendered_lumi_np = rendered_lumi.cpu().numpy()
    print(rendered_lumi_np.max())
    lumi_img = torch_render.visualize_lumi(rendered_lumi_np,setup)

    save_root = "test_lumi_diligent/"
    os.makedirs(save_root,exist_ok=True)

    for which_img in range(sample_num):
        tmp_img = lumi_img[which_img].copy()
        tmp_img = tmp_img / tmp_img.max()
        cv2.imwrite(save_root +"{}.png".format(which_img),tmp_img*255.0)