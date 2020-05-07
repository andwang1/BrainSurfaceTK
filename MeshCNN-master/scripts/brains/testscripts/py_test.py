import subprocess

print("starting test")
subprocess.call("./code_test.sh")
try:
    subprocess.call("./code_test.sh")
    print("test passed")
except:
    print("Test Failed")

