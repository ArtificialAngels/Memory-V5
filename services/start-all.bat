@echo off
REM See docs/scripts/core/memory_v5/services/start-all.md

REM Load Ikaros environment
call "%~dp0..\..\core\env\ikaros-env.bat"
if errorlevel 1 (
    echo [FATAL] Ikaros-environment\ikaros-env.bat failed.
    pause
    exit /b 1
)

echo ========================================
echo   Ikaros Memory - Services
echo ========================================
echo.

REM Start embedding service
echo [1/1] Starting embedding service (:%IKAROS_PORT_EMBEDDING%)...
start "Ikaros-Embedding" /MIN "%IKAROS_MEMORY_SERVICES%\start-embedding.bat"
timeout /t 3 /nobreak >nul

echo.
echo [OK] Embedding service started.
echo     - Embedding: http://127.0.0.1:%IKAROS_PORT_EMBEDDING%
echo     - LLM:       http://127.0.0.1:%IKAROS_PORT_LLAMA% (Hermes Agent)
echo.
echo To stop: taskkill /F /IM llama-server.exe
echo.

REM Wait for llama-server processes to keep running
:keepalive
timeout /t 3600 /nobreak >nul
goto :keepalive
