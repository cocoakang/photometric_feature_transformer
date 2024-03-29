@echo off
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::: task start                                                         :::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
SET MODEL_ROOT="D:/SIGA20_models/siga19_6/models/"
SET MODEL_FILE_NAME="model_state_600000.pkl"
SET NODE_NAME="linear_projection"
SET SAMPLE_VIEW_NUM=12
SET ALL_POS=0
@echo on
::[STEP1]get pattern
python getPattern_using_restore.py %MODEL_ROOT% %MODEL_FILE_NAME% %NODE_NAME% %SAMPLE_VIEW_NUM%
::
@echo off
SET PATTERN_FILE_NAME="W.bin"
::::::[STEP2]flip negative pattern
@echo on
python pattern_flipper.py %MODEL_ROOT% %SAMPLE_VIEW_NUM% %ALL_POS%
@echo off
SET PATTERN_FILE_NAME="W_flipped.bin"
::
::[STEP3]quantize pattern
@echo on
python pattern_quantizer.py %MODEL_ROOT% %SAMPLE_VIEW_NUM% %ALL_POS%
@echo off
::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::: task end                                                           :::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::***********************************************************************************************************************************

@echo on