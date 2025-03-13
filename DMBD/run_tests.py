#!/usr/bin/env python3
"""
Dynamic Markov Blanket Detection (DMBD) Framework Test Runner
===========================================================

This script runs all the tests for the DMBD framework, including:
1. Core component tests (framework modules)
2. Integration tests
3. Synthetic data generation tests
4. Visualization tests
5. Gridworld simulation and analysis tests

Results are saved to the output directory with detailed logging.
"""

import os
import sys
import unittest
import logging
import time
import datetime
import argparse
from pathlib import Path

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import test modules
from tests.test_markov_blanket import *
from tests.test_information_flow import *
from tests.test_cognitive_identification import *
from tests.test_synthetic_data import *
from tests.test_visualization import *
from tests.test_gridworld_dmbd import *  # Add the gridworld tests


def setup_logging(log_file: str):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('DMBD-Tests')


def setup_output_dirs():
    """Create output directories for test results."""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    data_dir = output_dir / 'data'
    data_dir.mkdir(exist_ok=True)
    
    return {
        'output': output_dir,
        'figures': figures_dir,
        'data': data_dir
    }


def generate_synthetic_datasets(output_dirs, logger):
    """Generate synthetic datasets for testing."""
    from tests.test_synthetic_data import generate_test_datasets
    
    logger.info("Generating synthetic datasets for testing...")
    datasets = generate_test_datasets(output_dir=output_dirs['data'])
    
    logger.info(f"Generated {len(datasets)} synthetic datasets:")
    for name, info in datasets.items():
        logger.info(f"  - {name}: shape={info['data'].shape}, stored at {info['path']}")
    
    return datasets


def run_tests(output_dirs, logger):
    """Run all DMBD tests and return results."""
    start_time = time.time()
    logger.info("Running DMBD framework tests...")
    
    # Create a test suite with all tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Core tests
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestMarkovBlanket))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestDynamicMarkovBlanket))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestInformationFlow))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestCognitiveIdentification))
    
    # Data generation tests
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestSyntheticData))
    
    # Visualization tests
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestVisualization))
    
    # Gridworld tests
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGaussianBlurGridworld))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGridworldMarkovAnalyzer))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGridworldIntegration))
    
    # Run the tests with a test runner that logs results
    test_result_file = output_dirs['output'] / 'test_results.txt'
    with open(test_result_file, 'w') as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        result = runner.run(test_suite)
    
    # Log overall results
    elapsed_time = time.time() - start_time
    logger.info(f"Tests completed in {elapsed_time:.2f} seconds")
    logger.info(f"Ran {result.testsRun} tests")
    logger.info(f"Results: {result.wasSuccessful()}")
    logger.info(f"  - Failures: {len(result.failures)}")
    logger.info(f"  - Errors: {len(result.errors)}")
    
    if len(result.failures) > 0 or len(result.errors) > 0:
        logger.error("Some tests failed. See test_results.txt for details.")
        for failure in result.failures:
            logger.error(f"FAILURE: {failure[0]}")
            logger.error(f"{failure[1]}")
        for error in result.errors:
            logger.error(f"ERROR: {error[0]}")
            logger.error(f"{error[1]}")
    
    return result


def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description='Run DMBD framework tests')
    parser.add_argument('--output-dir', default='output', help='Output directory for test results')
    parser.add_argument('--skip-synthetic-data', action='store_true', help='Skip synthetic data generation')
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = setup_output_dirs()
    
    # Set up logging
    log_file = output_dirs['output'] / 'test_log.txt'
    logger = setup_logging(str(log_file))
    
    logger.info("DMBD Test Suite started")
    logger.info(f"Output directory: {output_dirs['output']}")
    
    # Generate synthetic data
    if not args.skip_synthetic_data:
        datasets = generate_synthetic_datasets(output_dirs, logger)
    
    # Run tests
    result = run_tests(output_dirs, logger)
    
    logger.info("DMBD Test Suite completed")
    
    # Return an error code if any tests failed
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main()) 