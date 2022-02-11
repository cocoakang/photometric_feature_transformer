@echo off
SET SAVE_ROOT="D:/SIGA20_freshmeat/5_21_hodgepodge/"
SET TASK_NAME="no_need_to_fill_in"
SET MODEL_PATH[0]="D:/SIGA20_models/5_21_nosw_8x10/models/"
SET MODEL_PATH[1]="D:/SIGA20_models/5_21_nosw_12x6/models/"
SET MODEL_PATH[2]="D:/SIGA20_models/5_21_nosw_18x4/models/"
SET BLANK= 
SET ALL_PATHS=
SET "MODEL_NUM=0" 

:SymLoop 
if defined MODEL_PATH[%MODEL_NUM%] ( 
   ::call echo %%MODEL_PATH[%MODEL_NUM%]%% 
   call SET ALL_PATHS=%ALL_PATHS%%BLANK%%%MODEL_PATH[%MODEL_NUM%]%%
   set /a "MODEL_NUM+=1"
   GOTO :SymLoop 
)
@echo on

python pattern_merger_pro.py %TASK_NAME% %SAVE_ROOT% %MODEL_NUM% %ALL_PATHS%

@echo off
::clear model_path var
SET "MODEL_NUM=0" 
:SymLoopC 
if defined MODEL_PATH[%MODEL_NUM%] ( 
   SET MODEL_PATH[%MODEL_NUM%]=
   set /a "MODEL_NUM+=1"
   GOTO :SymLoopC
)
@echo on