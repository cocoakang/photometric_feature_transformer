import numpy as np
import os

wallet_root = "/Users/ross/CVPR21/torch_renderer/wallet_of_torch_renderer/"
config_root = wallet_root + "diligent_mv_pot2/"

os.makedirs(config_root,exist_ok=True)

with open(config_root +"cam_pos.bin","wb") as pf:
    cam_pos = np.zeros((3,),np.float32)
    cam_pos[2] = cam_pos[2] + 1500.0
    cam_pos.tofile(pf)

#######################

diligent_root = "/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/pot2PNG/view_01/"

light_directions = np.loadtxt(diligent_root +"light_directions.txt",delimiter=" ").astype(np.float32)

light_positions = light_directions * 1500.0 * 100.0 + 1500.0

light_normals = np.zeros_like(light_positions)
light_normals[:,2] = 1.0

lights = np.stack((light_positions,light_normals),axis=0)#(2,96,3)

lights.tofile(config_root+"lights.bin")

#######################

with open(config_root+"visualize_config_torch.bin","wb") as pf:
    img_size = np.array((8,12),np.int32)
    img_size.tofile(pf)

    light_id = np.arange(12*8,dtype=np.int32)
    light_row_idx = 7 - light_id % 8
    light_col_idx = np.concatenate((
        [
            np.ones((8,),np.int32)*5,
            np.ones((8,),np.int32)*4,
            np.ones((8,),np.int32)*3,
            np.ones((8,),np.int32)*2,
            np.ones((8,),np.int32)*1,
            np.ones((8,),np.int32)*0,
            np.ones((8,),np.int32)*6,
            np.ones((8,),np.int32)*7,
            np.ones((8,),np.int32)*8,
            np.ones((8,),np.int32)*9,
            np.ones((8,),np.int32)*10,
            np.ones((8,),np.int32)*11
        ]
    ),axis=0)
    light_visualize_idx = np.stack((light_col_idx,light_row_idx),axis=1)

    light_visualize_idx.tofile(pf)
