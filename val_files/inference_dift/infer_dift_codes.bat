@echo off

SET DATA_ROOT="D:/CVPR21_freshmeat/9_26_gmdift/billiard/"
SET MODEL_ROOT="D:/CVPR21_models/9_26_gmdift/models/"
SET MODEL_FILE_NAME="model_state_390000.pkl"

SET SAMPLE_VIEW_NUM=24
SET ROTATE_VIEW_NUM=24
SET MEASUREMENT_LEN=16
SET /A DIFT_CODE_LEN_G=7
SET /A DIFT_CODE_LEN_M=3
SET COLMAP_CODE_LEN=4
SET VIEW_CODE_LEN=128

python infer_dift_codes.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %SAMPLE_VIEW_NUM% %ROTATE_VIEW_NUM% %MEASUREMENT_LEN% %DIFT_CODE_LEN_G% %DIFT_CODE_LEN_M% %VIEW_CODE_LEN%

SET /A DIFT_CODE_LEN=(%DIFT_CODE_LEN_G%+%DIFT_CODE_LEN_M%)*3
python compact_dift_codes.py %DATA_ROOT% %ROTATE_VIEW_NUM% %DIFT_CODE_LEN% %COLMAP_CODE_LEN%

@echo on