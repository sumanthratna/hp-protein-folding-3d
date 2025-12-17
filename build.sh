#!/bin/bash
# Build script for 3D HP Protein Folding project

set -e

echo "Building 3D HP Protein Folding project..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    # Try python3.12 first, fall back to python3
    if command -v python3.12 &> /dev/null; then
        python3.12 -m venv venv
    else
        python3 -m venv venv
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Build complete!"
echo "To activate the virtual environment, run: source venv/bin/activate"


