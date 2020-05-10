import subprocess

__author__ = “Francis Rhys Ward”
__license__ = “MIT”

print("starting test")
try:
    subprocess.call("scripts/brains/regression_scan_age_test.sh")
    print("Regression with added global feature Test Passed")
except:
    print("Regression with added global feature Test Failed")

