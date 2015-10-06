@echo off
NET SESSION >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    ECHO Administrator PRIVILEGES Detected! 
) ELSE (
    ECHO NOT AN ADMIN!
    goto :end
)

mklink /D %~dp0\..\fast_rcnn_models "I:\Plateforme\Aware\General\Caffe\Models\fast_rcnn\fast_rcnn_models"

:end
