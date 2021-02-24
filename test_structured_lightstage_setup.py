import sys
import numpy as np
TORCH_RENDER_PATH = "../torch_renderer/"
sys.path.append(TORCH_RENDER_PATH)
import torch_render
import torch
import math
from torch_render import Setup_Config_Structured_Lightstage
import open3d as o3d

def draw_cam_frame(R_matrix,T_vec,dot_len=100.0,sample_num = 10):
    dot_length = np.linspace(0.0,dot_len,sample_num)
    point_collector = []
    color_collector = []

    cam_pos = -np.matmul(np.linalg.inv(R_matrix),T_vec)

    for which_len in range(sample_num):
        tmp_point = R_matrix * dot_length[which_len]
        tmp_point = tmp_point + cam_pos.T
        tmp_color = np.array([
            1.0,0.0,0.0,
            0.0,1.0,0.0,
            0.0,0.0,1.0
        ],np.float32).reshape((-1,3))
        point_collector.append(tmp_point)
        color_collector.append(tmp_color)
    point_collector = np.concatenate(point_collector,axis=0)
    color_collector = np.concatenate(color_collector,axis=0)

    return point_collector,color_collector


test_device = torch.device("cuda:0")

standard_rendering_parameters = {
        "config_dir":TORCH_RENDER_PATH+"wallet_of_torch_renderer/blackbox20_render_configs_1x1/"
    }
setup_input = Setup_Config_Structured_Lightstage(standard_rendering_parameters)

frame_num = 24

rotate_theta = torch.from_numpy(np.linspace(0.0,2*math.pi,frame_num,endpoint=False).reshape((-1,1)).astype(np.float32)).to(test_device)

R_matrix,T_vec = setup_input.get_rts(rotate_theta,test_device)

point_collector = []
color_collector = []
for which_angle in range(frame_num):
    tmp_frame_points,tmp_frame_colors = draw_cam_frame(R_matrix[which_angle].cpu().numpy(),T_vec[which_angle].cpu().numpy())
    if which_angle == 0:
        tmp_frame_colors = np.ones_like(tmp_frame_colors)
    elif which_angle == 1:
        tmp_frame_colors = np.zeros_like(tmp_frame_colors)

        
    point_collector.append(tmp_frame_points)
    color_collector.append(tmp_frame_colors)

point_collector = np.concatenate(point_collector,axis=0)
color_collector = np.concatenate(color_collector,axis=0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_collector)
pcd.colors = o3d.utility.Vector3dVector(color_collector)
o3d.io.write_point_cloud("cam_frames.ply", pcd)