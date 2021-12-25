## About

The official implementation of the paper [Learning Efficient Photometric Feature Transform for Multi-view Stereo](https://openaccess.thecvf.com/content/ICCV2021/html/Kang_Learning_Efficient_Photometric_Feature_Transform_for_Multi-View_Stereo_ICCV_2021_paper.html).

***Under construction...***

## Preparation

**torch_renderer**

**training data**

**multi channel COLMAP**

**other libraries**

## Training

1. run train_horizontal.sh to train Intensity-sensitive Branch
2. run train_vertical.sh to train Intensity-insensitive Branch
3. run train_combine.sh to load the pretrained model and jointly train the network

## Light stage Data


## [DiLiGenT-MV](https://sites.google.com/site/photometricstereodata/mv)


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