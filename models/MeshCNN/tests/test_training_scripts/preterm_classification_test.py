import subprocess
"""
Tests that training/testing scripts run correctly on a small subset   of data.
"""

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

print("starting test")

test_script_dir = "scripts/brains/test_scripts/"
try:
    subprocess.call(test_script_dir + "preterm_classification_test.sh")
    print("Preterm classification training test passed.")
except:
    print("Preterm classification training test failed.")

