## About

The official implementation of the paper [Learning Efficient Photometric Feature Transform for Multi-view Stereo](https://openaccess.thecvf.com/content/ICCV2021/html/Kang_Learning_Efficient_Photometric_Feature_Transform_for_Multi-View_Stereo_ICCV_2021_paper.html).

***Under construction...***

## Preparation

**torch_renderer**

Please clone this [repo](https://github.com/cocoakang/torch_renderer). As specified in tame.py(line 14), the repo should be in the same folder of training codes. The related calibration file can be found [here](https://drive.google.com/file/d/1TdN1woBJuuGFB4Ylai8pasNnNzG2NQES/view?usp=sharing).

**training data**

Download the pregenerated training data from [here](https://drive.google.com/drive/folders/19xyME8WgNMj6vIVtqK5hqBfhMsx5aUV9?usp=sharing). Extract the data to the folder specified in train_horizontal.sh. In tame.py, the training program will visualize feature space every 5000 steps, the related data(required by tame.py in line 270) can be found [here](https://drive.google.com/file/d/1JrucBvAYFcw_iwNjTF_HmYCO4c_o8ZsP/view?usp=sharing).

**multi channel COLMAP**

Please refer to this [repo](https://github.com/cocoakang/colmap_multichannel) to install the modified COLMAP for testing.

## Training

1. run train_horizontal.sh to train Intensity-sensitive Branch
2. run train_vertical.sh to train Intensity-insensitive Branch
3. run train_combine.sh to load the pretrained model and jointly train the network

## Testing
### Light stage Data
Please download the data and trained model from [here](https://www.aliyundrive.com/s/qzu24ZZ84Tf).
Use val_files/inference_dift/infer_dift_codes.bat to generate feature maps. A folder named feature_maps will be generated in the folder of each objects.
We provide pre-generated COLMAP dense folder for each objects, named as undistort_feature_dift.
Please undistort the multi-channel featuremaps and run dense reconstruction with modifed [multi-channel COLMAP](https://github.com/cocoakang/colmap_multichannel).

    colmap image_undistorter --image_path /path/to/feature/map/folder --input_path /path/to/undistort_feature_dif/ --output_path /path/to/undistort_feature_dif/ --input_type BIN

    colmap patch_match_stereo --workspace_path . --PatchMatchStereo.multi_channel 1 --PatchMatchStereo.geom_consistency 1 --PatchMatchStereo.sigma_spatial 15 --PatchMatchStereo.sigma_color 5.0 --PatchMatchStereo.num_samples 20 --PatchMatchStereo.ncc_sigma 1.0


### [DiLiGenT-MV](https://sites.google.com/site/photometricstereodata/mv)
Please checkout the brach diligent_mv and then follow the instruction in README.md.

License
---

Our source code is released under the GPL-3.0 license for acadmic purposes. The only requirement for using the code in your research is to cite our paper:

    @InProceedings{Kang_2021_ICCV,
        author    = {Kang, Kaizhang and Xie, Cihui and Zhu, Ruisheng and Ma, Xiaohe and Tan, Ping and Wu, Hongzhi and Zhou, Kun},
        title     = {Learning Efficient Photometric Feature Transform for Multi-View Stereo},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2021},
        pages     = {5956-5965}
    }

For commercial licensing options, please email hwu at acm.org.   
See COPYING for the open source license.
