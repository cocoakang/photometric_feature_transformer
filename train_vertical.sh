#!/bin/bash

data_root=../training_data/aniso_log_40_2b_with_rotate/

python tame.py $data_root --pretrained_model_pan_h "" --pretrained_model_pan_v "" --search_model --search_which geometry --m_len 3 --code_len 7 --start_seed 344191 --torch_manual_seed 9455306 --train_mine_seed 3150488 --val_mine_seed 1364108