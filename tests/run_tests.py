"""
Test runner script for QKD simulation tests.

This script runs all the tests for the QKD simulation application.
"""

import unittest
import sys
import os

def run_tests():
    """Run all tests for the QKD simulation application."""
    # Add the parent directory to the path so we can import the package
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return the number of failures and errors
    return len(result.failures) + len(result.errors)

if __name__ == "__main__":
    # Run the tests and exit with the appropriate status code
    sys.exit(run_tests())
