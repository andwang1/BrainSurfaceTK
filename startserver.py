import os
import sys
import time

SETTINGS = "GUI/BasicSite/settings.py"
MANAGE_PATH = "GUI/manage.py"
MEDIA_ROOT = "/vol/biomedic2/aa16914/shared/MScAI_brain_surface/media"


def modify_settings(set_debug_status_to=True, media_path=""):
    if set_debug_status_to:
        cboolv = False
    else:
        cboolv = True
    set_debug_status_to = str(set_debug_status_to).capitalize()
    cboolv = str(cboolv).capitalize()
    info = dict()
    with open(SETTINGS, 'r') as file:
        filedata = file.readlines()
    for i, line in enumerate(filedata):
        if "DEBUG" in line:
            filedata[i] = line.replace(cboolv, set_debug_status_to)
        if "MEDIA_URL" in line and line.strip(" ")[0] != "#":
            info["MEDIA_URL"] = line.split(" ")[-1].split('"')[-2][:-1]
        if "MEDIA_ROOT" in line and line.strip(" ")[0] != "#":
            filedata[i] = 'MEDIA_ROOT = ' + '\"' + str(media_path) + '\"' + "\n"
    with open(SETTINGS, 'w') as file:
        file.writelines(filedata)
    return info


if __name__ == "__main__":

    args = [arg for arg in sys.argv]

    os.system(" ".join(["python", MANAGE_PATH, "makemigrations", "--noinput"]))
    os.system(" ".join(["python", MANAGE_PATH, "migrate"]))

    if "imperial" not in args:
        MEDIA_ROOT = "GUI/media"

    if "prod" in args:
        info = modify_settings(False, MEDIA_ROOT)
        print(MEDIA_ROOT)
        print(os.getcwd())
        time.time()
        print("--url-alias", info["MEDIA_URL"], MEDIA_ROOT)
        os.system(r" ".join(["python", MANAGE_PATH, "collectstatic", "--noinput"]))
        os.system(r" ".join(
            ["python", MANAGE_PATH, "runmodwsgi", "--url-alias", info["MEDIA_URL"], MEDIA_ROOT, "--limit-request-body",
             "104857600", "--request-timeout 120"]))
    else:
        modify_settings(True, MEDIA_ROOT)
        os.system(" ".join(["python", MANAGE_PATH, "runserver"]))
