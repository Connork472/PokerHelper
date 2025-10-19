#!/bin/bash
# Setup script for PokerHelper virtual environment

echo "🎯 Setting up PokerHelper virtual environment..."

# Remove existing virtual environment
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating new virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Test tkinter
echo "Testing tkinter..."
python -c "import tkinter; print('✅ tkinter works!')"

# Test basic imports
echo "Testing basic imports..."
python -c "
import cv2
import numpy as np
import mss
import treys
print('✅ All basic dependencies work!')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use PokerHelper:"
echo "  source .venv/bin/activate"
echo "  python run.py"
echo ""
echo "Or run directly:"
echo "  python main.py    # GUI interface"
echo "  python cli_main.py # CLI interface"
