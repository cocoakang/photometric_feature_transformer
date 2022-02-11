import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

data_root = "D:/CVPR_DIFT/writing/tmp_mess/4lights/"
max_error = 30#degree
left_up = (150,72)
img_width = 230
img_height= 316

normal_gt = cv2.imread(data_root+"Normal_gt.png")[:,:,::-1]
normal_gt = normal_gt[left_up[1]:left_up[1]+img_height,left_up[0]:left_up[0]+img_width]
mask = np.where(normal_gt>0,np.ones_like(normal_gt),np.zeros_like(normal_gt))
normal_infered = cv2.imread(data_root+"18_Normal_CVPR14a.png")[:,:,::-1]
normal_infered = normal_infered[left_up[1]:left_up[1]+img_height,left_up[0]:left_up[0]+img_width]


normals = [normal_gt,normal_infered]

for which_normal in range(2):
    normals[which_normal] = normals[which_normal].astype(np.float32)/255.0
    normals[which_normal] = normals[which_normal] * 2.0 - 1.0

normal_gt,normal_infered = normals[0],normals[1]

normal_dot = np.sum(normal_gt*normal_infered,axis=2)
normal_dot = np.clip(normal_dot,-1.0,1.0)

normal_error = np.arccos(normal_dot)/math.pi*180.0/max_error
normal_error = np.clip(normal_error,0.0,1.0)

# cv2.imshow("normal_Error",normal_error/180.0)
# cv2.waitKey(0)

fig, ax = plt.subplots()
mycmap = plt.get_cmap("jet")

normal_error_trans = mycmap(normal_error)
normal_error_trans = normal_error_trans[:,:,:3]
normal_error_trans = normal_error_trans[:,:,::-1]
normal_error_trans = np.where(mask > 0, normal_error_trans,np.ones_like(normal_error_trans))
cv2.imwrite(data_root+"error.png",normal_error_trans*255.0)

im = ax.imshow(normal_error,cmap=mycmap,vmin=0, vmax=1)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax,ticks=np.linspace(0, 1.0, 3))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ticks = np.linspace(0,max_error/180.0*math.pi,3).tolist()
ticks = ["{:.2f}".format(a) for a in ticks]
cbar.ax.set_yticklabels(ticks)  # vertically oriented colorbar
cbar.ax.set_ylabel("", rotation=-90, va="bottom")
fig.tight_layout()
plt.axis('off')
# plt.show()
# plt.savefig(data_root+"error_map.png")

error_map = cv2.imread(data_root+"error_map.png")

left_up = (526,0)
img_width = 86 
img_height= 446
error_map = error_map[left_up[1]:left_up[1]+img_height,left_up[0]:left_up[0]+img_width]
error_map = np.concatenate([
    np.ones((0,img_width,3),np.uint8)*255,
    error_map,
    np.ones((80,img_width,3),np.uint8)*255
],axis=0)
cv2.imwrite(data_root+"error_bar.png",error_map)