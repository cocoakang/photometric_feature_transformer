SET DATA_ROOT="D:/CVPR21_freshmeat/9_16_old_4_best_vertical_grey/sphere/"
SET MODEL_ROOT="D:/CVPR21_models/9_16_old_4_best_vertical_grey/models/"
SET MODEL_FILE_NAME="model_state_180000.pkl"

SET SAMPLE_VIEW_NUM=24
SET ROTATE_VIEW_NUM=1
SET MEASUREMENT_LEN=4
SET DIFT_CODE_LEN=4
SET VIEW_CODE_LEN=128

REM python infer_dift_codes.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %SAMPLE_VIEW_NUM% %ROTATE_VIEW_NUM% %MEASUREMENT_LEN% %DIFT_CODE_LEN% %VIEW_CODE_LEN%
python compact_dift_codes.py %DATA_ROOT% %ROTATE_VIEW_NUM% %DIFT_CODE_LEN%