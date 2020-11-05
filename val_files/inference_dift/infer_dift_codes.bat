@echo off

SET DATA_ROOT="D:/CVPR21_freshmeat/11_4_sc1/billiard/"
SET MODEL_ROOT="D:/CVPR21_models/11_4_sc_m5_g3_md9_gd7/models/"
SET MODEL_FILE_NAME="model_state_270000.pkl"

SET SAMPLE_VIEW_NUM=24
SET ROTATE_VIEW_NUM=24
SET MEASUREMENT_LEN=8
SET /A DIFT_CODE_LEN_G=9
SET /A DIFT_CODE_LEN_M=7
SET COLMAP_CODE_LEN=4
SET VIEW_CODE_LEN=128

python infer_dift_codes.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %SAMPLE_VIEW_NUM% %ROTATE_VIEW_NUM% %MEASUREMENT_LEN% %DIFT_CODE_LEN_G% %DIFT_CODE_LEN_M% %VIEW_CODE_LEN%

SET /A DIFT_CODE_LEN=(%DIFT_CODE_LEN_G%+%DIFT_CODE_LEN_M%)*3
python compact_dift_codes.py %DATA_ROOT% %ROTATE_VIEW_NUM% %DIFT_CODE_LEN% %COLMAP_CODE_LEN%

@echo on