## Generate torch_render configs
Use generate_diligent_rendering_config.py to generate the configurations files including light position, light normal etc. One can use the generated configuration files to train the photometric feature transformer for a specific object in DiLiGent-mv dataset. 
<em>Note the lights and camera positions are different among objects, so one should generate configuration files for each object, instead of using the same files.</em>

## Transfer DiLiGenT-mv data to the input of this project
One can use the script <em>val_files/inference_dift/prepare_diligent_data.py</em> to generate input data for this project.

## Transfer input to feature vectors
With trained models, one can transfer the prepared data to feature vector. Two demo bash scripts are also provided to load our pretrained models: val_files/inference_dift/infer_dift_codes_bear4.sh and val_files/inference_dift/infer_dift_codes_bear_full.sh. The former one is show our model can still produce promising result even the number of input measurements is only 4. The other one is to show general result. We provided several model snaps of different training iterations. The featue space and sensitivity to infereflection are different.

## COLMAP reconstruction
To reconstruct point clouds with COLMAP, please copy the folder named undistort_feature_dift_bear to DiLiGenT-mv/mvpmsData/bearPNG/. Then use udt.sh to undistort input images, dense.sh to run patch_match_stereo and fuse.sh to fuse depth map and normal map to point clouds.
