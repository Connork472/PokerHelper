#!/bin/bash
# PokerHelper Launcher
# Automatically activates virtual environment and runs the application

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo ""
    echo "Please run the installation script first:"
    echo -e "  ${BLUE}./install.sh${NC}"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo -e "${RED}✗ Could not activate virtual environment${NC}"
    echo "The .venv directory exists but activation script is missing."
    echo "Try running: ./install.sh"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}✗ main.py not found!${NC}"
    echo "Please run this script from the PokerHelper directory."
    exit 1
fi

# Run the application
echo -e "${GREEN}🚀 Starting PokerHelper...${NC}"
echo ""

# Check if GUI is available, otherwise use CLI
if python3 -c "import tkinter" 2>/dev/null; then
    python3 main.py
else
    echo -e "${YELLOW}⚠ Tkinter not available, using CLI interface${NC}"
    echo ""
    python3 run.py
fi

