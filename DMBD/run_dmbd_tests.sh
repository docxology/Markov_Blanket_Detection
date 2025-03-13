#!/usr/bin/env bash

# DMBD Comprehensive Test Runner and Report Generator
# ===================================================
#
# This script runs the DMBD test suite and generates an HTML report
# with comprehensive visualizations of Dynamic Markov Blanket detection

# Set up error handling
set -e
trap 'echo "Error occurred at line $LINENO. Command: $BASH_COMMAND"' ERR

# Print header
echo "=========================================================="
echo "DMBD Comprehensive Test Suite with PyTorch Integration"
echo "=========================================================="
echo "Started at: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Create output directory if it doesn't exist
mkdir -p DMBD/output

# Check Python and torch availability
echo "Checking Python environment..."
python3 -c "import sys; print(f'Python version: {sys.version}')"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Function to measure execution time
start_time=$(date +%s)
function show_elapsed_time() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(( (elapsed % 3600) / 60 ))
    local seconds=$((elapsed % 60))
    echo "Elapsed time: ${hours}h ${minutes}m ${seconds}s"
}

# Execute test suite
echo "Running DMBD test suite..."
cd DMBD
python3 run_tests.py

# Check the exit status
if [ $? -ne 0 ]; then
    echo "Error: Test suite execution failed with exit code $?"
    show_elapsed_time
    exit 1
fi

echo "Test suite completed successfully."
echo ""

# Generate HTML report
echo "Generating HTML test report..."
python3 generate_report.py

# Check the exit status
if [ $? -ne 0 ]; then
    echo "Error: Report generation failed with exit code $?"
    show_elapsed_time
    exit 1
fi

echo "HTML report generated successfully."
echo ""

# Display summary information
echo "Test Summary:"
echo "-------------"
echo "Output directory: $(pwd)/output"
echo "Results summary: $(pwd)/output/test_summary.txt"
echo "HTML report: $(pwd)/output/dmbd_report.html"
echo "Total visualizations: $(find output/figures -name "*.png" | wc -l)"
echo ""

# Show elapsed time
show_elapsed_time
echo "Tests completed at: $(date)"
echo "=========================================================="

# Return to the original directory
cd ..

echo "To view the HTML report, open: $(pwd)/DMBD/output/dmbd_report.html" 