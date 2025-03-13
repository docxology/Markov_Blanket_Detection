#!/bin/bash
#
# Dynamic Markov Blanket Detection - Gridworld Analysis
# ====================================================
#
# This script runs a complete analysis of Dynamic Markov Blankets in a gridworld
# simulation and generates an HTML report with the results.
#

# Set working directory to the script location
cd "$(dirname "$0")"

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
GRID_SIZE="30 30"
TIME_POINTS=100
RADIUS=10.0
SIGMA=2.0
THRESHOLD=0.1
OUTPUT_DIR="output/gridworld_dmbd"
USE_TORCH=true
SKIP_ANIMATION=false
QUICK_MODE=false
INCLUDE_RAW_DATA=false

# Print banner
echo -e "\n${BLUE}================================================================================${NC}"
echo -e "${BLUE}    DYNAMIC MARKOV BLANKET DETECTION - GRIDWORLD ANALYSIS${NC}"
echo -e "${BLUE}================================================================================${NC}\n"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --grid-size)
      GRID_SIZE="$2 $3"
      shift 3
      ;;
    --time-points)
      TIME_POINTS="$2"
      shift 2
      ;;
    --radius)
      RADIUS="$2"
      shift 2
      ;;
    --sigma)
      SIGMA="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --use-torch)
      USE_TORCH=true
      shift
      ;;
    --skip-animation)
      SKIP_ANIMATION=true
      shift
      ;;
    --quick-mode)
      QUICK_MODE=true
      shift
      ;;
    --include-raw-data)
      INCLUDE_RAW_DATA=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --grid-size HEIGHT WIDTH   Size of the grid (default: 30 30)"
      echo "  --time-points NUM          Number of time points to simulate (default: 100)"
      echo "  --radius NUM               Radius of the circular path (default: 10.0)"
      echo "  --sigma NUM                Standard deviation of the Gaussian blur (default: 2.0)"
      echo "  --threshold NUM            Threshold for Markov blanket detection (default: 0.1)"
      echo "  --output-dir DIR           Directory to save outputs (default: output/gridworld_dmbd)"
      echo "  --use-torch                Use PyTorch for computations (default: true)"
      echo "  --skip-animation           Skip generating animations (default: false)"
      echo "  --quick-mode               Run a quicker analysis with smaller grid (default: false)"
      echo "  --include-raw-data         Include raw data in the HTML report (default: false)"
      echo "  --help                     Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Apply quick mode if requested
if [ "$QUICK_MODE" = true ]; then
  echo -e "${YELLOW}Running in quick mode with reduced parameters...${NC}"
  GRID_SIZE="15 15"
  TIME_POINTS=30
  OUTPUT_DIR="${OUTPUT_DIR}/quick_mode"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Grid size: $GRID_SIZE"
echo -e "  Time points: $TIME_POINTS"
echo -e "  Radius: $RADIUS"
echo -e "  Sigma: $SIGMA"
echo -e "  Threshold: $THRESHOLD"
echo -e "  Output directory: $OUTPUT_DIR"
echo -e "  Use PyTorch: $USE_TORCH"
echo -e "  Skip animations: $SKIP_ANIMATION"
echo -e "  Include raw data in report: $INCLUDE_RAW_DATA"

# Prepare command arguments
GRID_SIZE_ARGS=($GRID_SIZE)
TORCH_ARG=""
ANIMATION_ARG=""

if [ "$USE_TORCH" = true ]; then
  TORCH_ARG="--use-torch"
fi

if [ "$SKIP_ANIMATION" = true ]; then
  ANIMATION_ARG="--skip-animation"
fi

if [ "$INCLUDE_RAW_DATA" = true ]; then
  RAW_DATA_ARG="--include-raw-data"
else
  RAW_DATA_ARG=""
fi

# Start timing
START_TIME=$(date +%s)

echo -e "\n${BLUE}Step 1: Running gridworld DMBD analysis...${NC}"
python3 run_gridworld_analysis.py \
  --grid-size "${GRID_SIZE_ARGS[0]}" "${GRID_SIZE_ARGS[1]}" \
  --time-points "$TIME_POINTS" \
  --radius "$RADIUS" \
  --sigma "$SIGMA" \
  --threshold "$THRESHOLD" \
  --output-dir "$OUTPUT_DIR" \
  $TORCH_ARG $ANIMATION_ARG

# Check if analysis was successful
if [ $? -ne 0 ]; then
  echo -e "\n${RED}Error: Gridworld DMBD analysis failed!${NC}"
  exit 1
fi

echo -e "\n${BLUE}Step 2: Generating HTML report...${NC}"
python3 generate_gridworld_report.py \
  --analysis-dir "$OUTPUT_DIR" \
  --output-file "$OUTPUT_DIR/report.html" \
  --title "Dynamic Markov Blanket Analysis - Gridworld Simulation" \
  $RAW_DATA_ARG

# Check if report generation was successful
if [ $? -ne 0 ]; then
  echo -e "\n${RED}Error: HTML report generation failed!${NC}"
  exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

echo -e "\n${GREEN}Analysis and report generation completed successfully in ${ELAPSED_TIME} seconds!${NC}"
echo -e "Results are available in: ${OUTPUT_DIR}"
echo -e "HTML report: ${OUTPUT_DIR}/report.html"

# Suggest next steps
echo -e "\n${BLUE}Next steps:${NC}"
echo -e "  - Open the HTML report to view the results"
echo -e "  - Explore the generated visualizations and animations"
echo -e "  - Modify parameters to analyze different gridworld configurations"

exit 0 