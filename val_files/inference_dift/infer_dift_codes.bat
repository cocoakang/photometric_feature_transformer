SET DATA_ROOT="D:/CVPR21_freshmeat/7_13_l2posnormal/sphere/"
SET MODEL_ROOT="D:/CVPR21_models/7_13_l2posnormal/models/"
SET MODEL_FILE_NAME="model_state_50000.pkl"

SET SAMPLE_VIEW_NUM=24
SET ROTATE_VIEW_NUM=24
SET MEASUREMENT_LEN=4
SET DIFT_CODE_LEN=8
SET VIEW_CODE_LEN=128

python infer_dift_codes.py %DATA_ROOT% %MODEL_ROOT% %MODEL_FILE_NAME% %SAMPLE_VIEW_NUM% %ROTATE_VIEW_NUM% %MEASUREMENT_LEN% %DIFT_CODE_LEN% %VIEW_CODE_LEN%
python compact_dift_codes.py %DATA_ROOT% %ROTATE_VIEW_NUM% %DIFT_CODE_LEN%