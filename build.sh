#!/bin/bash
# Build script for 3D HP Protein Folding project
# CPSC 5740 Final Project - Sumanth Ratna (sr2437)

set -e

echo "=========================================="
echo "Building 3D HP Protein Folding Project"
echo "=========================================="
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    # Try python3.12 first, then python3
    if command -v python3.12 &> /dev/null; then
        echo "Using Python 3.12"
        python3.12 -m venv venv
    elif command -v python3 &> /dev/null; then
        echo "Using Python 3"
        python3 -m venv venv
    else
        echo "Error: Python 3 not found. Please install Python 3."
        exit 1
    fi
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
echo "Installing dependencies..."
# on Zoo, CUDA 12.6:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install matplotlib

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "To run the project:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run the test script: ./test.sh"
echo ""
