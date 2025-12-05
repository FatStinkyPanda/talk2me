#!/bin/bash
# Talk2Me Setup Script for Linux/macOS
# Sets up Python virtual environment, installs dependencies, and downloads ML models

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    echo "Talk2Me Setup Script"
    echo "==================="
    echo

    # Check Python availability
    log_info "Checking Python installation..."
    if command_exists python3; then
        PYTHON_CMD="python3"
        log_success "Found python3"
    elif command_exists python; then
        PYTHON_CMD="python"
        log_success "Found python"
    else
        log_error "Python not found. Please install Python 3.8+ first."
        exit 1
    fi

    # Verify Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        log_error "Python $PYTHON_VERSION found, but Python 3.8+ is required."
        exit 1
    fi
    log_success "Python version: $PYTHON_VERSION"

    # Check if virtual environment already exists
    if [ -d "venv" ]; then
        log_warning "Virtual environment already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf venv
        else
            log_info "Using existing virtual environment."
        fi
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            log_error "Failed to create virtual environment."
            exit 1
        fi
        log_success "Virtual environment created."
    fi

    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
    if [ $? -ne 0 ]; then
        log_error "Failed to activate virtual environment."
        exit 1
    fi

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    if [ $? -ne 0 ]; then
        log_warning "Failed to upgrade pip, continuing..."
    fi

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt not found in current directory."
        exit 1
    fi

    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        log_error "Failed to install dependencies."
        exit 1
    fi
    log_success "Dependencies installed successfully."

    # Check if download script exists
    if [ ! -f "scripts/download_models.py" ]; then
        log_error "scripts/download_models.py not found."
        exit 1
    fi

    # Run model download script
    log_info "Downloading ML models..."
    python scripts/download_models.py
    if [ $? -ne 0 ]; then
        log_error "Failed to download models."
        exit 1
    fi

    # Deactivate virtual environment
    deactivate

    echo
    log_success "Setup completed successfully!"
    echo
    echo "To activate the virtual environment in future sessions:"
    echo "  source venv/bin/activate"
    echo
    echo "To run the application:"
    echo "  source venv/bin/activate"
    echo "  python -m src.talk2me.api.main"
    echo
    echo "Happy talking! 🎤"
}

# Run main function
main "$@"
