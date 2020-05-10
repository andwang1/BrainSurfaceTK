import subprocess

__author__ = “Francis Rhys Ward”
__license__ = “MIT”

print("starting test")
try:
    subprocess.call("scripts/brains/regression_test.sh")
    print("Regression Test Passed")
except:
    print("Regression Test Failed")

