import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

data_root = "/Users/ross/CVPR21_freshmeat/DiLiGenT-MV/mvpmsData/view_01/"

normal_gt = cv2.imread(data_root+"Normal_gt.png")[:,:,::-1]
normal_infered = cv2.imread(data_root+"normal_infered.png")[:,:,::-1]

normals = [normal_gt,normal_infered]

for which_normal in range(2):
    normals[which_normal] = normals[which_normal].astype(np.float32)/255.0
    normals[which_normal] = normals[which_normal] * 2.0 - 1.0

normal_gt,normal_infered = normals[0],normals[1]

normal_dot = np.sum(normal_gt*normal_infered,axis=2)
normal_dot = np.clip(normal_dot,-1.0,1.0)

normal_error = np.arccos(normal_dot)/math.pi*45.0

# cv2.imshow("normal_Error",normal_error/180.0)
# cv2.waitKey(0)

fig, ax = plt.subplots()
im = ax.imshow(normal_error,cmap=plt.get_cmap("jet"))

# We want to show all ticks...
# ax.set_xticks(np.arange(len(farmers)))
# ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
# ax.set_xticklabels(farmers)
# ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax,ticks=np.arange(0, 45, 5))
cbar.ax.set_ylabel("", rotation=-90, va="bottom")
fig.tight_layout()
plt.show()