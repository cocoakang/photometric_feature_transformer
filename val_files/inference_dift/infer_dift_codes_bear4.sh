#!/bin/bash

DATA_ROOT="/home/cocoa_kang/dift_freshmeat/DiLiGenT-MV/mvpmsData/bearPNG/"
MODEL_ROOT="../../runs/diligent_global_local_bear_structured_correct_newtest_4/models/"
MODEL_FILE_NAME="model_state_1650000.pkl"

SAMPLE_VIEW_NUM=20
ROTATE_VIEW_NUM=20
MEASUREMENT_LEN=4
DIFT_CODE_LEN_G=64
DIFT_CODE_LEN_M=64
COLMAP_CODE_LEN=4

python infer_dift_codes.py $DATA_ROOT $MODEL_ROOT $MODEL_FILE_NAME $SAMPLE_VIEW_NUM $ROTATE_VIEW_NUM $MEASUREMENT_LEN $DIFT_CODE_LEN_G $DIFT_CODE_LEN_M

DIFT_CODE_LEN=$(( 3 * $(( $DIFT_CODE_LEN_G + $DIFT_CODE_LEN_M)) ))
python compact_dift_codes.py $DATA_ROOT $ROTATE_VIEW_NUM $DIFT_CODE_LEN $COLMAP_CODE_LEN
