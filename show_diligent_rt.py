import numpy as np
import open3d as o3d
import scipy.io as scio
import os

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

data_root = "/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/pot2PNG/"
config_root = "/Users/ross/CVPR21/torch_renderer/wallet_of_torch_renderer/diligent_mv_pot2/"
os.makedirs(config_root,exist_ok=True)

origin_data = scio.loadmat(data_root+"Calib_Results.mat")

intrinsic = origin_data["KK"]

np.savetxt(data_root+"intrinsic.txt",intrinsic,delimiter=",")


cam_pos_collector = []
pf = open(data_root+"cam_pos.txt","w")

point_collector = []
color_collector = []
R_matrix_collector = []
T_vec_collector = []
for which_view in range(20):
    R_matrix = origin_data["Rc_{}".format(which_view+1)]
    T_vec = origin_data["Tc_{}".format(which_view+1)]

    R_matrix_collector.append(R_matrix)
    T_vec_collector.append(T_vec)

    cam_pos = -np.matmul(np.linalg.inv(R_matrix),T_vec)
    cam_pos_collector.append(cam_pos)
    
    pf.write("view {:02d}: ({:.3f},{:.3f},{:.3f})\n".format(which_view+1,cam_pos[0][0],cam_pos[1][0],cam_pos[2][0]))

    tmp_frame_points,tmp_frame_colors = draw_cam_frame(R_matrix,T_vec)
    
    point_collector.append(tmp_frame_points)
    color_collector.append(tmp_frame_colors)

pf.close()

point_collector = np.concatenate(point_collector,axis=0)
color_collector = np.concatenate(color_collector,axis=0)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_collector)
pcd.colors = o3d.utility.Vector3dVector(color_collector)
o3d.io.write_point_cloud(data_root+"cam_frames.ply", pcd)

print(cam_pos_collector[0])
cam_pos_collector[0].astype(np.float32).tofile(config_root+"cam_pos.bin")

##############
R_matrix = R_matrix_collector[0]
T_Vector = T_vec_collector[0]

light_directions = np.loadtxt(data_root +"view_01/light_directions.txt",delimiter=" ").astype(np.float32)

light_positions = light_directions * 1500.0 * 100.0#in cam frame

light_normals = np.zeros_like(light_positions)
light_normals[:,2] = 1.0#(in cam frame)

light_positions[:,1] = -light_positions[:,1]
light_positions[:,2] = -light_positions[:,2]

light_normals[:,1] = -light_normals[:,1]
light_normals[:,2] = -light_normals[:,2]


light_positions = light_positions.T#(3,lightnum)
light_positions = np.matmul(np.linalg.inv(R_matrix),(light_positions-T_vec))#(3,lightnum)
light_normals = light_normals.T#(3,lightnum)
light_normals = np.matmul(np.linalg.inv(R_matrix),light_normals)#(3,lightnum)
light_normals = light_normals / np.linalg.norm(light_normals,axis=0,keepdims=True)

light_positions = light_positions.T#(lightnum,T)
# print("light_positions:\n",light_positions)
light_normals = light_normals.T#(lightnum,T)
# print("light_normals:\n",light_normals)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(light_positions)
light_color = np.repeat(np.expand_dims(np.linspace(0.0,1.0,num=12*8),axis=1),3,axis=1)
pcd.colors = o3d.utility.Vector3dVector(light_color)
o3d.io.write_point_cloud(config_root+"light_pos.ply", pcd)


lights = np.stack((light_positions,light_normals),axis=0)#(2,96,3)

lights.astype(np.float32).tofile(config_root+"lights.bin")

#############

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
