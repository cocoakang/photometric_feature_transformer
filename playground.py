import torch
import math
import numpy as np
import matplotlib.pyplot as plt

root = "/home/cocoa_kang/training_tasks/current_work/CVPR21_DIFT/BRDF_feature_extract/logs/sig20_init/details/params_init/"

#############################
name_base = "0"
tf_data = np.fromfile(root+"{}.bin".format(name_base),np.float32)
tc_data = np.fromfile(root+"{}_tc.bin".format(name_base),np.float32)

# tf_data = tf_data.reshape([3,24576,4])
# tc_data = tc_data.reshape([4,24576,3])

print(tf_data.max())
print(tc_data.max())
print(tf_data.min())
print(tc_data.min())
print(tf_data[:10])
print(tc_data[:10])