#!/usr/bin/env python3
"""
Gridworld DMBD Test Runner
========================

This script runs the tests for the Gridworld Dynamic Markov Blanket detection module.
It ensures proper import paths are set up before running the tests.
"""

import os
import sys
import unittest
import importlib.util

# Add the current directory to the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the test module
test_file = os.path.join(current_dir, 'tests', 'test_gridworld_dmbd.py')
test_module = import_module_from_file('test_gridworld_dmbd', test_file)

# Get the test classes
TestGaussianBlurGridworld = getattr(test_module, 'TestGaussianBlurGridworld')
TestGridworldMarkovAnalyzer = getattr(test_module, 'TestGridworldMarkovAnalyzer')
TestGridworldIntegration = getattr(test_module, 'TestGridworldIntegration')


def run_tests():
    """Run the gridworld DMBD tests."""
    print("Running Gridworld DMBD tests...")
    
    # Create a test suite with gridworld tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add gridworld tests
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGaussianBlurGridworld))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGridworldMarkovAnalyzer))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestGridworldIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 