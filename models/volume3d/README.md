## Applying CNN to volumetric data

#### Steps for instructions

SPLIT
in utils/ place names.pk

META in utils
,participant_id,session_id,gender,birth_age,birth_weight,singleton,scan_age,scan_number,radiology_score,sedation,exist
0,CC00549XX22,157600,Female,42.0,3.685,Single,42.142857142857,1,2,0.0,1

CHANGE PATH_TO_DATA in utils/models.py; hence place anywhere u like

FILE naming 
sub-{patient_id}_ses-{session_id}_T2w_graymatter.nii.gz 