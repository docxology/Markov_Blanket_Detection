#!/bin/bash
#
# Install dependencies for the Gridworld DMBD Analysis
# ===================================================
#
# This script installs all the required Python packages for the gridworld
# Dynamic Markov Blanket analysis.
#

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${BLUE}    INSTALLING DEPENDENCIES FOR GRIDWORLD DMBD ANALYSIS${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 not found. Please install Python and pip first.${NC}"
    exit 1
fi

# Create and activate virtual environment (optional)
if [[ "$1" == "--venv" ]]; then
    echo -e "${YELLOW}Creating and activating virtual environment...${NC}"
    
    # Check if venv module is available
    if ! python3 -c "import venv" &> /dev/null; then
        echo -e "${RED}Error: Python venv module not found. Please install it first.${NC}"
        echo -e "On Ubuntu/Debian: sudo apt install python3-venv"
        exit 1
    fi
    
    # Create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate venv
    source venv/bin/activate
    
    echo -e "${GREEN}Virtual environment activated.${NC}"
fi

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"

# Core packages
pip3 install numpy pandas matplotlib tqdm

# PyTorch (CPU-only version by default)
if [[ "$1" == "--cuda" ]] || [[ "$2" == "--cuda" ]]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip3 install torch torchvision torchaudio
else
    echo -e "${YELLOW}Installing PyTorch (CPU-only)...${NC}"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Additional packages for visualization
pip3 install seaborn networkx

# Check if ffmpeg is installed (required for animations)
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}Warning: ffmpeg not found. Animations may not work.${NC}"
    echo -e "On Ubuntu/Debian: sudo apt install ffmpeg"
fi

echo -e "\n${GREEN}All dependencies installed successfully!${NC}"
echo -e "You can now run the gridworld DMBD analysis:"
echo -e "  ./run_gridworld_dmbd.sh"
echo -e "  ./run_gridworld_dmbd.sh --quick-mode    # For a quick test run"

# If using virtual environment, print reminder
if [[ "$1" == "--venv" ]]; then
    echo -e "\n${YELLOW}Remember to activate the virtual environment before running:${NC}"
    echo -e "  source venv/bin/activate"
fi

exit 0 