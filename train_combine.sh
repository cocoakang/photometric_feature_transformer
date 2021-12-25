#!/bin/bash

data_root=../training_data/aniso_log_40_2b_with_rotate/
pretrained_model_pan_h=runs/learn_l2_ml5_mg0_dla0_dlna9_dg0/models/model_state_90000.pkl
pretrained_model_pan_v=runs/learn_l2_ml0_mg3_dla0_dlna0_dg7/models/model_state_90000.pkl

python tame.py $data_root --pretrained_model_pan_h $pretrained_model_pan_h --pretrained_model_pan_v $pretrained_model_pan_v