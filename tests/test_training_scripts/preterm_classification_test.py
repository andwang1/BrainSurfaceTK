import subprocess
 """
 Tests that training/testing scripts run correctly on a small subset   of data.
 """

 print("starting test")

 test_script_dir = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/ rhys/deepl_brain_surfaces/MeshCNN-master/scripts/brains/test_scripts/ "
 try:
     subprocess.call(test_script_dir +                                 "preterm_classification_test.sh")
     print("Preterm classification training test passed.")
 except:
     print("Preterm classification training test failed.")

