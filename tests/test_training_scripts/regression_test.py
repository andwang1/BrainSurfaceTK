import subprocess

print("starting test")
try:
    subprocess.call("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/deepl_brain_surfaces/MeshCNN-master/scripts/brains/regression_test.sh")
    print("Regression Test Passed")
except:
    print("Regression Test Failed")

