import os
import sys

SETTINGS = "GUI/BasicSite/settings.py"
MANAGE_PATH = "GUI/manage.py"
MEDIA_PATH = "GUI/media"


def set_debug_to(boolv=True):
    if boolv:
        cboolv = False
    else:
        cboolv = True
    boolv = str(boolv).capitalize()
    cboolv = str(cboolv).capitalize()
    info = dict()
    with open(SETTINGS, 'r') as file:
        filedata = file.readlines()
    for i, line in enumerate(filedata):
        if "DEBUG" in line:
            filedata[i] = line.replace(cboolv, boolv)
        if "MEDIA_URL" in line:
            info["MEDIA_URL"] = line.split(" ")[-1].split('"')[-2][:-1]
    with open(SETTINGS, 'w') as file:
        file.writelines(filedata)
    return info


if __name__ == "__main__":

    args = [arg for arg in sys.argv]

    os.system(" ".join(["python", MANAGE_PATH, "makemigrations", "--noinput"]))
    os.system(" ".join(["python", MANAGE_PATH, "migrate"]))

    if "prod" in args:
        info = set_debug_to(False)
        os.system(r" ".join(["python", MANAGE_PATH, "collectstatic", "--noinput"]))
        os.system(r" ".join(["python", MANAGE_PATH, "runmodwsgi", "--url-alias", info["MEDIA_URL"], MEDIA_PATH, "--limit-request-body", "104857600"]))
    else:
        set_debug_to(True)
        os.system(" ".join(["python", MANAGE_PATH, "runserver"]))
