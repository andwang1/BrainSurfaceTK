import os
import random
import string
import time

from backend.evaluate_pointnet_regression import predict_age
from backend.evaluate_pointnet_segmentation import segment as brain_segment
from django.conf import settings
from nibabel import load as nib_load
from nilearn.plotting import view_img as ni_view_img

from .models import Session


def get_unique_session(session_id, participant_id):
    result = Session.objects.filter(session_id=session_id, participant_id=participant_id).defer('uploaded')
    if result.count() == 1:
        return result.get(), None
    elif result.count() == 0:
        return None, "ERROR: No search were found in the database!"
    else:
        return None, "ERROR: Multiple searches were found, search is not unique!"


def build_session_table(record):
    record_dict = vars(record)

    # TODO: Rewrite to exclude clutter
    field_names = record_dict.keys()
    table_names = list()
    table_values = list()
    for field_name in field_names:
        # This removes field names that we don't want to be displayed to the user.
        if field_name.startswith("_") or field_name == "id" or field_name.endswith("file") \
                or field_name.endswith("path") or "uploaded" in field_name.lower():
            continue
        tmp_value = record_dict[field_name]
        if isinstance(tmp_value, float):
            tmp_value = round(tmp_value, 3)
        elif tmp_value == "":
            tmp_value = "NA"
        table_names.append(field_name.replace("_", " ").title().replace("Id", "ID"))
        table_values.append(tmp_value)
    return table_names, table_values


def get_mri_js_html(record):
    mri_file = record.mri_file
    if mri_file.name != "":
        if os.path.isfile(mri_file.path) & (mri_file.path.endswith("nii") or mri_file.path.endswith("nii.gz")):
            img = nib_load(mri_file.path)
            mri_js_html = ni_view_img(img, colorbar=False, bg_img=False, black_bg=True, cmap='gray')
            return mri_js_html, None
        return None, "ERROR: Either MRI file doesn't exist or doesn't end with .nii!"
    return None, None


def get_surf_file_url(record):
    surf_file = record.surface_file
    if surf_file.name != "":
        if os.path.isfile(surf_file.path) & surf_file.path.endswith("vtp"):
            return surf_file.url, None
        return None, "ERROR: Either Surface file doesn't exist or doesn't end with .vtp!"
    return None, None


def pointnet_run_prediction(file_url):
    return round(predict_age(os.path.join(settings.MEDIA_ROOT, file_url.strip(settings.MEDIA_URL))), 3)


def pointnet_run_segmentation(file_url):
    abs_file_path = os.path.join(settings.MEDIA_ROOT, file_url.strip(settings.MEDIA_URL))

    # Check if tmp folder exists
    tmp_path = os.path.join(settings.MEDIA_ROOT, "tmp/")
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    current_tmp_files = [f for f in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, f))]
    while True:
        tmp_file_name = random_string(stringLength=15) + ".vtp"
        if tmp_file_name not in current_tmp_files:
            break

    tmp_file_path = brain_segment(brain_path=abs_file_path,
                                  folder_path_to_write=tmp_path,
                                  tmp_file_name=tmp_file_name)

    # Segmented Temporary File URL
    return os.path.join(settings.MEDIA_URL,tmp_file_path.split(settings.MEDIA_ROOT)[-1][1:])


def random_string(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(stringLength))


def remove_file(relative_file_path, allocated_file_life=10):
    if relative_file_path is not None:
        file_path = os.path.join(settings.MEDIA_ROOT, relative_file_path.strip(settings.MEDIA_URL))
        if os.path.exists(file_path):
            file_life_time = os.path.getmtime(file_path)
            if time.time() - file_life_time < allocated_file_life:
                time.sleep(allocated_file_life - (time.time() - file_life_time))
            os.remove(file_path)
            return True
    return False
