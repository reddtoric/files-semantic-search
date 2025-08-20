@echo off
REM Windows batch file for semantic file search
REM Usage: search.bat "your search query"
REM Example: search.bat "machine learning code"

setlocal EnableDelayedExpansion

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if file_search.py exists
if not exist "file_search.py" (
    echo ❌ Error: file_search.py not found in current directory
    echo Please run this from the directory containing the search tool
    pause
    exit /b 1
)

REM Check if arguments provided
if "%~1"=="" (
    echo 🔍 Semantic File Search - Windows Launcher
    echo ================================================
    echo Usage: search.bat "your search query" [options]
    echo.
    echo Examples:
    echo   search.bat "machine learning code"
    echo   search.bat "database connection" --model fast
    echo   search.bat "authentication logic" --cache --debug
    echo.
    echo For full help: python file_search.py --help
    echo.
    pause
    exit /b 0
)

REM Run the search with all provided arguments
echo 🔍 Running semantic file search...
echo Query: %1
echo.

python file_search.py %*

REM Check exit code
if errorlevel 1 (
    echo.
    echo ❌ Search completed with errors
) else (
    echo.
    echo ✅ Search completed successfully
)

REM Pause only if running interactively (double-clicked)
echo %cmdcmdline% | find /i "%~0" >nul
if not errorlevel 1 (
    echo.
    echo Press any key to close...
    pause >nul
)
