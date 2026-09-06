@echo off
REM See docs/scripts/core/memory_v5/services/start-llm.md

REM Load Ikaros environment (sets IKAROS_PYTHON / IKAROS_LLAMA_SERVER / IKAROS_MEMORY ...)
call "%~dp0..\..\core\env\ikaros-env.bat"
if errorlevel 1 (
    echo [FATAL] Ikaros-environment\ikaros-env.bat failed.
    pause
    exit /b 1
)

REM Resolve the local LLM launch command dynamically from models/model_config.json.
REM First run scans the model dir and persists the choice; no model name is hardcoded here.
set "RESOLVER=%IKAROS_MEMORY%\models\model_config.py"
if not exist "%RESOLVER%" (
    echo [FATAL] model resolver not found: %RESOLVER%
    pause
    exit /b 1
)

"%IKAROS_PYTHON%" "%RESOLVER%" --emit-bat > "%TEMP%\ikaros_llm_launch.tmp.bat"
if errorlevel 1 (
    echo [FATAL] model resolver failed to emit launch command.
    pause
    exit /b 1
)

echo [Ikaros Memory] Launching local LLM via models/model_config.json ...
call "%TEMP%\ikaros_llm_launch.tmp.bat"
set "LAUNCH_RC=%errorlevel%"
del /q "%TEMP%\ikaros_llm_launch.tmp.bat" >nul 2>&1
exit /b %LAUNCH_RC%
