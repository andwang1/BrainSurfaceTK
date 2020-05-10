__author__ = “Francis Rhys Ward”
__license__ = “MIT”

def write_seseg(eseg_path, seseg_path, patient_id, ses_id):
    eseg_file = eseg_path + patient_id +"_"+ses_id+".eseg"
    seseg_file = seseg_path + patient_id +"_"+ses_id+".seseg"
    with open(eseg_file) as f:
        eseg = f.read().splitlines()

    labels = range(38)

    with open(seseg_file, 'w') as f:
        for label in eseg:
            row = [0 if l is not int(label) else 1 for l in labels]
            f.write(str(row).strip("[]").replace(",", ""))
            f.write("\n")
