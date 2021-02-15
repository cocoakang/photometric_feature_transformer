from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",default="/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/readingPNG/")

    args = parser.parse_args()

    #load gt
    gt = o3d.io.read_point_cloud(args.data_root+"mesh_Gt.ply")

    gt_points = np.asarray(gt.points)

    #load ours
    ours = o3d.io.read_point_cloud(args.data_root+"poissoned.ply")
    ours_points = np.asarray(ours.points)

    #evaluate here
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ours_points)
    distances, indices = nbrs.kneighbors(gt_points)

    distances = np.clip(distances,0.0,2.0)
    distances = distances / 2.0

    cmap=plt.get_cmap("jet")
    tmp_color = cmap(distances)
    tmp_color = np.reshape(tmp_color,(-1,4))[:,:3]
    print(tmp_color)
    print(distances.shape)
    print(tmp_color.shape)
    gt.colors = o3d.utility.Vector3dVector(tmp_color)
    

    o3d.visualization.draw_geometries([gt])

