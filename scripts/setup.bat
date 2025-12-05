@echo off
REM Talk2Me Setup Script for Windows
REM Sets up Python virtual environment, installs dependencies, and downloads ML models

setlocal enabledelayedexpansion

REM Colors for output (Windows CMD)
REM Note: Windows CMD has limited color support, using simple text

:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:main
echo Talk2Me Setup Script
echo ====================
echo.

REM Check Python availability
call :log_info "Checking Python installation..."
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    call :log_success "Found python"
) else (
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=py
        call :log_success "Found py"
    ) else (
        call :log_error "Python not found. Please install Python 3.8+ first."
        exit /b 1
    )
)

REM Verify Python version
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %PYTHON_VERSION% | findstr /r "3\.[8-9] 3\.[1-9][0-9]" >nul
if %errorlevel% neq 0 (
    call :log_error "Python %PYTHON_VERSION% found, but Python 3.8+ is required."
    exit /b 1
)
call :log_success "Python version: %PYTHON_VERSION%"

REM Check if virtual environment already exists
if exist "venv" (
    call :log_warning "Virtual environment already exists."
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        call :log_info "Removing existing virtual environment..."
        rmdir /s /q venv
    ) else (
        call :log_info "Using existing virtual environment."
    )
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    call :log_info "Creating Python virtual environment..."
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        call :log_error "Failed to create virtual environment."
        exit /b 1
    )
    call :log_success "Virtual environment created."
)

REM Activate virtual environment
call :log_info "Activating virtual environment..."
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    call :log_error "Failed to activate virtual environment."
    exit /b 1
)

REM Upgrade pip
call :log_info "Upgrading pip..."
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    call :log_warning "Failed to upgrade pip, continuing..."
)

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    call :log_error "requirements.txt not found in current directory."
    exit /b 1
)

REM Install Python dependencies
call :log_info "Installing Python dependencies..."
pip install -r requirements.txt
if %errorlevel% neq 0 (
    call :log_error "Failed to install dependencies."
    exit /b 1
)
call :log_success "Dependencies installed successfully."

REM Check if download script exists
if not exist "scripts\download_models.py" (
    call :log_error "scripts\download_models.py not found."
    exit /b 1
)

REM Run model download script
call :log_info "Downloading ML models..."
python scripts\download_models.py
if %errorlevel% neq 0 (
    call :log_error "Failed to download models."
    exit /b 1
)

REM Deactivate virtual environment
call deactivate

echo.
call :log_success "Setup completed successfully!"
echo.
echo To activate the virtual environment in future sessions:
echo   venv\Scripts\activate.bat
echo.
echo To run the application:
echo   venv\Scripts\activate.bat
echo   python -m src.talk2me.api.main
echo.
echo Happy talking! 🎤
goto :eof

REM Run main function
call :main
