#!/bin/bash
# PokerHelper Installation Script
# Automatically sets up virtual environment and installs all dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  PokerHelper - Automated Setup${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check if Python 3 is installed
check_python() {
    print_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed!"
        echo ""
        echo "Please install Python 3 first:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install python3"
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo "  sudo apt-get install python3 python3-venv python3-pip"
        else
            echo "  Visit https://www.python.org/downloads/"
        fi
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION found"
}

# Detect OS and handle OS-specific requirements
handle_os_requirements() {
    print_info "Detecting operating system..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_success "Detected macOS"
        
        # Check for tkinter (required for GUI)
        if ! python3 -c "import tkinter" 2>/dev/null; then
            print_warning "Tkinter not available. GUI may not work."
            echo "To install Tkinter on macOS:"
            echo "  brew install python-tk"
            echo ""
            echo "Or use the CLI interface instead: python cli_main.py"
        else
            print_success "Tkinter is available"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_success "Detected Linux"
        
        # Check for tkinter
        if ! python3 -c "import tkinter" 2>/dev/null; then
            print_warning "Tkinter may not be installed"
            echo "Install with: sudo apt-get install python3-tk"
        else
            print_success "Tkinter is available"
        fi
    else
        print_warning "Unknown OS: $OSTYPE"
    fi
}

# Create virtual environment
create_venv() {
    print_info "Setting up virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning ".venv already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf .venv
            python3 -m venv .venv
            print_success "Virtual environment created"
        else
            print_info "Using existing virtual environment"
        fi
    else
        python3 -m venv .venv
        print_success "Virtual environment created"
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_error "Could not find virtual environment activation script"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    # Upgrade pip first
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install requirements
    print_info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    
    print_success "All dependencies installed"
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."
    
    local failed=0
    
    # Test critical imports
    print_info "Testing critical imports..."
    
    python3 -c "import treys" 2>/dev/null && print_success "treys ✓" || { print_error "treys ✗"; failed=1; }
    python3 -c "import cv2" 2>/dev/null && print_success "opencv-python ✓" || { print_error "opencv-python ✗"; failed=1; }
    python3 -c "import numpy" 2>/dev/null && print_success "numpy ✓" || { print_error "numpy ✗"; failed=1; }
    python3 -c "import mss" 2>/dev/null && print_success "mss ✓" || { print_error "mss ✗"; failed=1; }
    python3 -c "import PIL" 2>/dev/null && print_success "pillow ✓" || { print_error "pillow ✗"; failed=1; }
    python3 -c "import sklearn" 2>/dev/null && print_success "scikit-learn ✓" || { print_error "scikit-learn ✗"; failed=1; }
    python3 -c "import onnx" 2>/dev/null && print_success "onnx ✓" || { print_error "onnx ✗"; failed=1; }
    
    if [ $failed -eq 1 ]; then
        print_error "Some dependencies failed to import"
        print_warning "Try running: pip install -r requirements.txt"
        return 1
    fi
    
    print_success "All critical dependencies verified"
    return 0
}

# Ensure launcher script exists and is executable
create_launcher() {
    print_info "Setting up launcher script..."
    
    # Check if start.sh exists, if not create it
    if [ ! -f "start.sh" ]; then
        print_info "Creating launcher script..."
        cat > start.sh << 'EOF'
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
EOF
        print_success "Launcher script created (start.sh)"
    else
        print_info "Launcher script already exists"
    fi
    
    # Ensure it's executable
    chmod +x start.sh
    print_success "Launcher script is ready (start.sh)"
}

# Make scripts executable
make_executable() {
    print_info "Making scripts executable..."
    chmod +x install.sh 2>/dev/null || true
    print_success "Scripts are executable"
}

# Main installation flow
main() {
    print_header
    
    check_python
    handle_os_requirements
    create_venv
    activate_venv
    install_dependencies
    
    if verify_installation; then
        create_launcher
        make_executable
        
        echo ""
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  ✓ Installation Complete!${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
        echo ""
        echo "To start PokerHelper, run:"
        echo -e "  ${BLUE}./start.sh${NC}"
        echo ""
        echo "Or manually:"
        echo -e "  ${BLUE}source .venv/bin/activate${NC}"
        echo -e "  ${BLUE}python3 main.py${NC}"
        echo ""
    else
        print_error "Installation completed with warnings"
        echo "Some dependencies may not be working correctly."
        echo "Please check the error messages above."
        exit 1
    fi
}

# Run main function
main

