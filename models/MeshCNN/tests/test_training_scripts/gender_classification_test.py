import subprocess

__author__ = "Francis Rhys Ward"
__license__ = "MIT"

"""
Tests that training/testing scripts run correctly on a small subset of data.
"""

print("starting test")

test_script_dir = "scripts/brains/test_scripts/"
try:
    subprocess.call(test_script_dir + "meshcnn_gender_classification_training_test.sh")
    print("Gender classification training test passed.")
except:
    print("Gender classification training test failed.")
