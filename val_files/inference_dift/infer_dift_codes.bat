@echo off

SET DATA_ROOT="D:/CVPR21_freshmeat/9_17_grey2/9_17_old_4_best_vertical_grey2/"
SET MODEL_ROOT="D:/CVPR21_models/9_17_old_4_best_vertical_grey2/models/"
SET MODEL_FILE_NAME="model_state_90000.pkl"

SET SAMPLE_VIEW_NUM=24
SET ROTATE_VIEW_NUM=24
SET MEASUREMENT_LEN=4
SET /A DIFT_CODE_LEN=4
SET COLMAP_CODE_LEN=4
SET VIEW_CODE_LEN=128

python infer_dift_codes.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %SAMPLE_VIEW_NUM% %ROTATE_VIEW_NUM% %MEASUREMENT_LEN% %DIFT_CODE_LEN% %VIEW_CODE_LEN%

SET /A DIFT_CODE_LEN=%DIFT_CODE_LEN%*3
python compact_dift_codes.py %DATA_ROOT% %ROTATE_VIEW_NUM% %DIFT_CODE_LEN% %COLMAP_CODE_LEN%

@echo on