import os
from csv import reader as csv_reader

from django.conf import settings

from .models import Session


def load_original_data(reset_upload_database, batch_size=64):
    """
    Wipes the SessionDatabase and optionally wipes the UploadedSessionDatabase before loading
     the original dataset into the Django builtin database & detects their associated files
    :return: redirects to homepage with messages to notify the user for success or any errors
    """
    # Clear each database here
    Session.objects.all().filter(uploaded=False).delete()
    if reset_upload_database == 'on':
        Session.objects.all().filter(uploaded=True).delete()

    tsv_path = os.path.join(settings.MEDIA_ROOT, settings.ORIGINAL_META_DATA_PATH)
    mri_path = os.path.join(settings.MEDIA_ROOT, settings.ORIGINAL_MRI_DATA_PATH)
    vtp_path = os.path.join(settings.MEDIA_ROOT, settings.ORIGINAL_VTP_DATA_PATH)

    if not os.path.isfile(tsv_path):
        # return {"success": False, "message": "Either this is not a file or the location is wrong!"}
        return {"success": False, "message": tsv_path}

    # Check if the tsv file exists
    expected_ordering = ['participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                         'scan_age', 'scan_number', 'radiology_score', 'sedation']

    found_mri_file_names = [f for f in os.listdir(mri_path) if f.endswith("nii") or f.endswith("nii.gz")]
    found_vtps_files_names = [f for f in os.listdir(vtp_path) if f.endswith("vtp")]

    with open(tsv_path) as foo:
        reader = csv_reader(foo, delimiter='\t')
        sessions = list()
        for i, row in enumerate(reader):
            if i == 0:
                if row != expected_ordering:
                    return {"success": False,
                            "message": "FAILED! The table column names aren't what was expected or in the wrong order."}
                continue

            elif i % batch_size == 0:
                Session.objects.bulk_create(sessions, batch_size=batch_size)
                sessions = list()

            (participant_id, session_id, gender, birth_age, birth_weight, singleton, scan_age,
             scan_number, radiology_score, sedation) = row

            mri_file_path = next((os.path.join(settings.ORIGINAL_MRI_DATA_PATH, x)
                                  for x in found_mri_file_names if (participant_id and session_id) in x), "")

            surface_file_path = next((os.path.join(settings.ORIGINAL_VTP_DATA_PATH, x)
                                      for x in found_vtps_files_names if (participant_id and session_id) in x), "")

            # Check for session ID uniqueness
            if Session.objects.all().filter(session_id=session_id, participant_id=participant_id).count() > 0:
                print(f'tsv contains non-uniques: {session_id}, {participant_id}')
                continue

            sessions.append(Session(participant_id=participant_id,
                                    session_id=int(session_id),
                                    gender=gender,
                                    birth_age=float(birth_age),
                                    birth_weight=float(birth_weight),
                                    singleton=singleton,
                                    scan_age=float(scan_age),
                                    scan_number=int(scan_number),
                                    radiology_score=radiology_score,
                                    sedation=sedation,
                                    uploaded=False,
                                    mri_file=mri_file_path,
                                    surface_file=surface_file_path))

    return {"success": True, "message": "SUCCESS"}
