#!/bin/bash

data_root=../training_data/aniso_log_40_2b_with_rotate/

python tame.py $data_root --pretrained_model_pan_h "" --pretrained_model_pan_v "" --search_model --search_which material --m_len 5 --code_len 9