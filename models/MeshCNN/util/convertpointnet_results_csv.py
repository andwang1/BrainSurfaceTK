__author__ = "Andy Wang"
__license__ = "MIT"

with open("presentation_results/results.csv", "r") as f:
    file_contents = f.readlines()

epoch_counter = 1

base = "presentation_results/test_logs/scan_age/pointnet/testacc_full_log_"

for line in file_contents:
    if "Val" in line:
        f = open(f"{base}{epoch_counter}.csv", "w")
        continue
    if "Epoch average" in line:
        f.close()
        epoch_counter += 1
        continue
    elements = line.split(",")
    unique_id = elements[0] + "_" + elements[1]
    del elements[0]
    elements[0] = unique_id
    f.write(",".join(elements))
