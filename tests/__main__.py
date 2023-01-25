# This file is used to initialize the tests package.
# It is required for unittest to discover and run tests inside the test folder.

import unittest

# Discover all tests in the test folder and add them to the test suite.
test_suite = unittest.TestLoader().discover('tests', pattern='*_test.py')

if __name__ == "__main__":
    unittest.TextTestRunner().run(test_suite)