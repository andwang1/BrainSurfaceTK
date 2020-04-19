import subprocess
"""
Tests that training/testing scripts run correctly on a small subset of data.
"""

print("starting test")

test_script_dir = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/rhys/deepl_brain_surfaces/MeshCNN-master/scripts/brains/test_scripts/"
try:
    subprocess.call(test_script_dir + "meshcnn_gender_classification_training_test.sh")
    print("Gender classification training test passed.")
except:
    print("Gender classification training test failed.")
