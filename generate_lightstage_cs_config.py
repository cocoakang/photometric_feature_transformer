import numpy as np
import os
import cv2
import shutil

from_root = "/Users/ross/CVPR21/torch_renderer/wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
save_root = "/Users/ross/CVPR21/torch_renderer/wallet_of_torch_renderer/blackbox20_render_configs_1x1_cs/"
calib_root = "/Users/ross/CVPR21/device_configuration/2020_11_2/"

os.makedirs(save_root,exist_ok=True)

extrinsic_file = cv2.FileStorage(calib_root+"extrinsic0.yml", cv2.FILE_STORAGE_READ)
rvec = np.asarray(extrinsic_file.getNode("rvec").mat())
tvec = np.asarray(extrinsic_file.getNode("tvec").mat())
rotM = np.asarray(cv2.Rodrigues(rvec)[0])

for file_name in ["mat_for_normal.bin","mat_model.bin","visualize_config_torch.bin","visualize_idxs.bin"]:
    shutil.copyfile(from_root+file_name,save_root+file_name)

cam_pos = np.fromfile(from_root+"cam_pos.bin",np.float32).reshape((3,1))
cam_pos = np.matmul(rotM,cam_pos)+tvec
cam_pos = np.zeros_like(np.reshape(cam_pos,(-1)))

cam_pos.astype(np.float32).tofile(save_root+"cam_pos.bin")

#####
lights = np.fromfile(from_root+"lights.bin",np.float32).reshape((2,-1,3))
light_pos = lights[0]
light_normals = lights[1]

light_pos = np.matmul(rotM,light_pos.T)+tvec
light_pos = light_pos.T

light_normals = np.matmul(rotM,light_normals.T)
light_normals = light_normals.T

with open(save_root+"lights.bin","wb") as pf:
    light_pos.astype(np.float32).tofile(pf)
    light_normals.astype(np.float32).tofile(pf)