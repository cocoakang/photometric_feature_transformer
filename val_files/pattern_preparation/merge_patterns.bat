@echo off
SET SAVE_ROOT="D:/CVPR21_freshmeat/9_26_gmdift/"
SET TASK_NAME="no_need_to_fill_in"
SET SAMPLE_VIEW_NUM=24
SET MODEL_PATH[0]="D:/CVPR21_models/9_26_gmdift/models/"
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

python pattern_merger.py %TASK_NAME% %SAVE_ROOT% %MODEL_NUM% %ALL_PATHS% %SAMPLE_VIEW_NUM%

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