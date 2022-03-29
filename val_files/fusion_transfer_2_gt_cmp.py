import numpy as np
import argparse
import open3d as o3d
import subprocess
from subprocess import Popen
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="H:/iccv_data/")
    parser.add_argument("--path_to_colmap",default="D:/SIG20/colmaps/colmap_orgin/colmap_build/__install__/")

    args = parser.parse_args()

    pf_output = open("scores_obj.txt","w")
    pf_output.write("id,c,a,\n")

    for obj_name in ["cat","fox","teacup","chicken"]:
        data_root = args.data_root+"{}/".format(obj_name)
        base_args =[
            args.path_to_colmap+"COLMAP.bat",
            "stereo_fusion",
            "--workspace_path",
            "H:/iccv_data/{}/undistort_feature_dift/".format(obj_name),
            "--output_path",
            "H:/iccv_data/{}/undistort_feature_dift/fused.ply".format(obj_name),
            ]

        # fusion
        the_process = Popen(
            base_args
        )
        exit_code = the_process.wait()
        print("exit_code:{}".format(exit_code))

        #load transform matrix
        trans_matrix = np.loadtxt(data_root+"trans_mat_2_gt.txt",delimiter=',')

        #load reconstructed point clouds
        pcd_load = o3d.io.read_point_cloud(data_root + "undistort_feature_dift/fused.ply")

        points = np.asarray(pcd_load.points)
        normals = np.asarray(pcd_load.normals)
        colors = np.asarray(pcd_load.colors)

        points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
        normals = np.concatenate([normals,np.zeros((normals.shape[0],1))],axis=-1)

        #trans here
        points = np.matmul(trans_matrix,points.T).T
        # points = points / points[:,[3]]
        points = points[:,:3]
        normals = np.matmul(trans_matrix,normals.T).T
        normals = normals[:,:3]

        #save
        pcd_load.points = o3d.utility.Vector3dVector(points.astype(np.float32))
        pcd_load.normals = o3d.utility.Vector3dVector(normals.astype(np.float32))
        o3d.io.write_point_cloud(data_root+"undistort_feature_dift/fused_in_world_space.ply", pcd_load,compressed=True)

        #The open3d saved file uses double, while ETH3DMultiViewEvaluation requires float data. Use meshlab to automatically transfer data format.
        the_process = Popen(
            [
                "meshlabserver",
                "-i",
                data_root+"undistort_feature_dift/fused_in_world_space.ply",
                "-o",
                data_root+"undistort_feature_dift/fused_in_world_space.ply"
            ]
        )
        exit_code = the_process.wait()
        print("exit_code:{}".format(exit_code))

        #check quality
        the_process = Popen(
            [
                "ETH3DMultiViewEvaluation.exe",
                "--reconstruction_ply_path",
                data_root+"undistort_feature_dift/fused_in_world_space.ply",
                "--ground_truth_mlp_path",
                data_root+"gt.mlp",
                "--tolerances",
                "1.0"
            ],
            stdout=subprocess.PIPE
        )
        exit_code = the_process.wait()
        print("exit_code:{}".format(exit_code))
        output, err = the_process.communicate()
        output = str(output, encoding = "utf-8").split("\n")
        c = float(output[5].strip("\n").split(": ")[1])
        a = float(output[6].strip("\n").strip("\n").split(": ")[1])
        print("{} {}".format(c,a))
        pf_output.write("{},{},{},\n".format(obj_name,c,a))
        pf_output.flush()

    pf_output.close()