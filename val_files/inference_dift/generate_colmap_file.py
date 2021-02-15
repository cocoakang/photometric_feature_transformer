import numpy as np
import open3d as o3d
import scipy.io as scio
import os
import argparse
from scipy.spatial.transform import Rotation as R

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/readingPNG/")
    parser.add_argument("--template_root",default="/Users/ross/CVPR21_freshmeat/")
    parser.add_argument("--image_num",type=int,default=20)

    args = parser.parse_args()

    ###################

    calib_data = scio.loadmat(args.data_root+"Calib_Results.mat")

    intrinsic = calib_data["KK"]

    save_root = args.data_root+"undistort_feature_dift/"
    os.makedirs(save_root,exist_ok=True)

    ##################load points
    label_point_root = args.data_root+"full_on_udt/marked_points/"
    point_num  = len([a for a in os.listdir(label_point_root) if '.txt' in a and "points" not in a])
    points_fitted = np.loadtxt(label_point_root+"points.txt",delimiter=',')

    feature_collector = [[] for i in range(args.image_num)]
    point_collector = []

    for which_point in range(point_num):
        tmp_data = np.loadtxt(label_point_root + "{}.txt".format(which_point),delimiter=',').astype(np.int32)
        tmp_point = {}
        tmp_point["point_id"] = which_point
        tmp_point["xyz"] = "{} {} {}".format(points_fitted[which_point][0],points_fitted[which_point][1],points_fitted[which_point][2])
        tmp_point["rgb"] = "255 255 255"
        tmp_point["error"] = "0.0"
        tmp_point["track"] = []

        feature_counter = 0
        for which_view,x,y in tmp_data:
            feature_collector[which_view].append((x,y,which_point))
            tmp_point["track"].append((which_view+1,feature_counter))
            feature_counter += 1
        
        point_collector.append(tmp_point)

    track_per_point = np.mean([len(a["track"]) for a in point_collector])
    feature_per_image = np.mean([len(a) for a in feature_collector])

    ########camera.txt
    pf_template = open(args.template_root+"cameras.txt","r")
    with open(save_root+"cameras.txt","w") as pf:
        pf.write(pf_template.readline())
        pf.write(pf_template.readline())
        pf.write(pf_template.readline())
        pf.write("1 OPENCV 612 512 {:.2f} {:.2f} {:.2f} {:.2f} 0 0 0 0\n".format(intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2]))
    
    pf_template.close()

    ########images.txt
    pf_template = open(args.template_root+"images.txt","r")
    with open(save_root+"images.txt","w") as pf:
        pf.write(pf_template.readline())
        pf.write(pf_template.readline())
        pf.write(pf_template.readline())
        pf.write("# Number of images: {}, mean observations per image: {:.4f}\n".format(args.image_num,feature_per_image))
        for which_image in range(args.image_num):
            R_matrix = calib_data["Rc_{}".format(which_image+1)]
            T_vec = calib_data["Tc_{}".format(which_image+1)].reshape((-1,))

            r = R.from_matrix(R_matrix)
            q = r.as_quat()

            pf.write("{} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} 1 pd_predicted_{}_0.png\n".format(which_image+1,q[3],q[0],q[1],q[2],T_vec[0],T_vec[1],T_vec[2],which_image))

            ##point
            for x,y,pointid in feature_collector[which_image]:
                pf.write("{} {} {} ".format(x,y,pointid))
            pf.write("\n")



    pf_template.close()


    #######points3D.txt
    pf_template = open(args.template_root+"points3D.txt","r")
    with open(save_root+"points3D.txt","w") as pf:
        pf.write(pf_template.readline())
        pf.write(pf_template.readline())
        pf.write("# Number of points: {}, mean track length: {}\n".format(point_num,track_per_point))
        
        #TODO
        for a_point in point_collector:
            pf.write("{} {} {} {}".format(a_point["point_id"],a_point["xyz"],a_point["rgb"],a_point["error"]))
            for view_id,feature_id in a_point["track"]:
                pf.write(" {} {}".format(view_id,feature_id))
            pf.write("\n")

    pf_template.close()


