import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import scipy.io as scio
import os
import open3d as o3d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/buddhaPNG/")
    parser.add_argument("--view_num",type=int,default=20)

    args = parser.parse_args()

    calib_mat = scio.loadmat(args.data_root+"Calib_Results.mat")

    intrinsic = torch.from_numpy(calib_mat["KK"].astype(np.float32))
    intrinsic = torch.unsqueeze(intrinsic,dim=0)
    rt_list = []
    for which_view in range(args.view_num):
        R_matrix = calib_mat["Rc_{}".format(which_view+1)]
        T_vec = calib_mat["Tc_{}".format(which_view+1)]

        M = np.vstack((np.hstack((R_matrix, T_vec)), [0, 0, 0 ,1]))

        M = torch.from_numpy(M.astype(np.float32))

        rt_list.append(M)
    
    rt_list = torch.stack(rt_list,dim=0)#(viewnum,4,4)

    label_point_root = args.data_root+"full_on_udt/marked_points/"
    point_num  = len([a for a in os.listdir(label_point_root) if '.txt' in a])

    loss_module = torch.nn.L1Loss()
    
    point_collector = []

    for which_point in range(point_num):
        label_data = np.loadtxt(label_point_root+"{}.txt".format(which_point),delimiter=',').astype(np.float32)

        visible_views = label_data[:,0].astype(np.int32)
        visible_view_num = label_data.shape[0]

        labeled_xy = torch.from_numpy(label_data[:,1:3])

        rts = rt_list[visible_views]
        tmp_intrinsic = intrinsic.repeat(visible_view_num,1,1)

        #overfitting here...
        point_position = torch.nn.Parameter(data = torch.from_numpy(np.array((0.0,0.0,0.0),np.float32)),requires_grad=True)#.to("cuda:0")
        lr = 1e-1# if train_configs["training_mode"] == "pretrain" else 1e-5
        optimizer = optim.Adam([point_position], lr=lr)


        for which_itr in range(10000):
            optimizer.zero_grad()

            tmp_point_position = torch.cat((point_position,torch.ones(1,dtype=point_position.dtype)),dim=0)

            tmp_point_position = tmp_point_position.reshape((1,4,1)).repeat(visible_view_num,1,1)

            tmp_point_position = torch.matmul(rts,tmp_point_position)#(visible_view_num,4,1)

            projected_xy = torch.matmul(tmp_intrinsic,tmp_point_position[:,:3,:])#(visible_view_num,3,1)
            projected_xy = projected_xy / (1e-5+projected_xy[:,[2],:])#(visible_view_num,3,1)

            infered_xy = projected_xy[:,:2,0]

            l2_loss = loss_module(infered_xy,labeled_xy)
            # print("labelxy:{} infered_xy:{} loss:{} point:{}".format(labeled_xy[0],infered_xy[0],l2_loss.item(),point_position))
            l2_loss.backward()
            optimizer.step()

        
        print("labelxy:{} infered_xy:{} loss:{} point:{}".format(labeled_xy[0],infered_xy[0],l2_loss.item(),point_position))
        point_collector.append(point_position.data.numpy().copy())
        print(point_collector)
    points = np.stack(point_collector,axis=0)
    np.savetxt(args.data_root+"full_on_udt/marked_points/points.txt",points,delimiter = ',')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(args.data_root+"optimized_pcd.ply", pcd)


