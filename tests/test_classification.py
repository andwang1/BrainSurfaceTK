import subprocess

print("starting test")
try:
    subprocess.call("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/deepl_brain_surfaces/MeshCNN-master/scripts/brains/code_test.sh")
    print("Test Passed")
except:
    print("Test Failed")
