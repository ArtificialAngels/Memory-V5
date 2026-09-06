@echo off
REM See docs/scripts/core/memory_v5/services/start-embedding.md
REM
REM Thin wrapper (2026-08-20 收敛为 ikaros 启动器, see docs/ikaros-launcher-design.md)
REM 真实实现: core/ikarosctl.py 的 embed 子命令

set "IKAROS_ROOT=%~dp0..\..\.."
for %%i in ("%IKAROS_ROOT%") do set "IKAROS_ROOT=%%~fi"

rem Self-check anchor -- fail fast if resolution is wrong
if not exist "%IKAROS_ROOT%\bin\ikaros-env.bat" (
    echo [FATAL] IKAROS_ROOT anchor failed: %IKAROS_ROOT%\bin\ikaros-env.bat not found
    exit /b 1
)

call "%IKAROS_ROOT%\bin\ikaros-env.bat"
if errorlevel 1 (
    echo [FATAL] ikaros-env.bat failed.
    exit /b 1
)

"%IKAROS_ROOT%\bin\ikaros.bat" embed %*
exit /b %ERRORLEVEL%
