import csv
import os

import warnings

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import permission_required
from django.contrib.auth.forms import AuthenticationForm
from django.http import JsonResponse
from django.shortcuts import render, redirect

from backend.evaluate_pointnet_regression import predict_age
from backend.evaluate_pointnet_segmentation import segment as brain_segment
from .forms import NewUserForm, UploadFileForm
from .models import Option, SessionDatabase, UploadedSessionDatabase

from nibabel import load as nib_load
from nilearn.plotting import view_img as ni_view_img

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "/main/static/main/data")


def homepage(request):
    if Option.objects.count() == 0:
        Option.objects.create(name="Look-up".title(), summary="Look-up session IDs".capitalize(), slug="lookup")
        Option.objects.create(name="Upload".title(), summary="Upload session ID".capitalize(), slug="upload")
        Option.objects.create(name="About".title(), summary="About this project".capitalize(), slug="about")
    options = Option.objects.all()
    return render(request, "main/homepage.html", context={"options": options})


def about(request):
    return render(request, "main/about.html")


def view_session_results(request, session_id=None):
    if request.method == "GET":
        for database in [SessionDatabase, UploadedSessionDatabase]:
            search_results = database.objects.filter(session_id=session_id)
            if search_results.count() is 1:
                break
            elif search_results.count() > 1:
                messages.error(request, "ERROR: Multiple records were found, session id is not unique!")
                return redirect("main:lookup")
        else:
            messages.error(request, "ERROR: No records were found, this session id was not found in the database!")
            return redirect("main:lookup")

        record = search_results.get()
        record_dict = vars(record)
        # TODO: Rewrite to exclude clutter
        field_names = record_dict.keys()
        table_names = list()
        table_values = list()
        for field_name in field_names:
            if field_name.startswith("_") or field_name == "id" or field_name.endswith("file") \
                or field_name.endswith("path"):
                continue
            tmp_value = record_dict[field_name]
            if isinstance(tmp_value, float):
                tmp_value = round(tmp_value, 3)
            elif tmp_value == "":
                tmp_value = "NA"
            table_names.append(field_name.replace("_", " ").title().replace("Id", "ID"))
            table_values.append(tmp_value)

        mri_file = record.mri_file
        mri_js_html = None
        if mri_file.name != "":
            if os.path.isfile(mri_file.path) & mri_file.path.endswith("nii"):
                # img = nib_load(mri_file.path)
                # mri_js_html = ni_view_img(img, colorbar=False, bg_img=False, black_bg=True, cmap='gray')
                mri_js_html = None
            else:
                messages.error(request, "ERROR: Either MRI file doesn't exist or doesn't end with .nii!")

        surf_file = record.surface_file
        surf_file_url = None
        if surf_file.name != "":
            surf_file_url = surf_file.url
            if not (os.path.isfile(surf_file.path) & surf_file.path.endswith("vtp")):
                surf_file_url = None
                messages.error(request, "ERROR: Either Surface file doesn't exist or doesn't end with .vtp!")

        return render(request, "main/results.html",
                      context={"session_id": session_id, "table_names": table_names, "table_values": table_values,
                               "mri_js_html": mri_js_html, "surf_file_url": surf_file_url,
                               'debug': search_results.get()})


@permission_required('admin.can_add_log_entry')
def load_data(request):
    if request.method == "POST":
        # Clear each database here
        SessionDatabase.objects.all().delete()
        reset_upload_database = request.POST.get("reset_upload_database", 'off')
        if reset_upload_database == 'on':
            UploadedSessionDatabase.objects.all().delete()

        # Check if the tsv file exists
        if not os.path.isfile(SessionDatabase().tsv_path):
            messages.error(request, "Either this is not a file or the location is wrong!")
            return redirect("main:load_database")

        expected_ordering = ['participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                             'scan_age', 'scan_number', 'radiology_score', 'sedation']

        found_mri_file_names = [f for f in os.listdir(SessionDatabase().default_mri_path) if f.endswith("nii")]
        found_vtps_files_names = [f for f in os.listdir(SessionDatabase().default_vtps_path)
                                  if f.endswith("vtp") & f.find("inflated") != -1 & f.find("hemi-L") != -1]

        with open(SessionDatabase().tsv_path) as foo:
            reader = csv.reader(foo, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    if row != expected_ordering:
                        messages.error(request, "FAILED! The table did not have the expected column name ordering.")
                        return redirect("main:load_database")
                    continue

                (participant_id, session_id, gender, birth_age, birth_weight, singleton, scan_age,
                 scan_number, radiology_score, sedation) = row

                mri_file_path = next((f"{os.path.join(SessionDatabase().default_mri_path, x)}"
                                      for x in found_mri_file_names if (participant_id and session_id) in x), "")

                surface_file_path = next((f"{os.path.join(SessionDatabase().default_vtps_path, x)}"
                                          for x in found_vtps_files_names if (participant_id and session_id) in x), "")

                if SessionDatabase.objects.all().filter(session_id=session_id).count() > 0:
                    print(f'tsv contains non-uniques session id: {session_id}')
                    continue

                SessionDatabase.objects.create(participant_id=participant_id,
                                               session_id=int(session_id),
                                               gender=gender,
                                               birth_age=float(birth_age),
                                               birth_weight=float(birth_weight),
                                               singleton=singleton,
                                               scan_age=float(scan_age),
                                               scan_number=int(scan_number),
                                               radiology_score=radiology_score,
                                               sedation=sedation,
                                               mri_file=mri_file_path,
                                               surface_file=surface_file_path)

                if mri_file_path != "":
                    record = SessionDatabase.objects.get(session_id=session_id)
                    record.mri_file.name = record.mri_file.name.split(settings.MEDIA_URL)[-1]
                    record.save()

                if surface_file_path != "":
                    record = SessionDatabase.objects.get(session_id=session_id)
                    record.surface_file.name = record.surface_file.name.split(settings.MEDIA_URL)[-1]
                    record.save()

        messages.success(request, "Successfully loaded data!")
        return redirect("main:homepage")

    return render(request, "main/load_database.html")


def lookup(request):
    if request.user.is_superuser:
        if request.method == "GET":
            session_id = request.GET.get("selected_session_id", None)
            if session_id is not None:
                return redirect("main:session_id_results", session_id=session_id, permanent=True)

            session_ids = [int(session.session_id) for session in SessionDatabase.objects.all()]
            uploaded_session_ids = [int(session.session_id) for session in UploadedSessionDatabase.objects.all()]

            return render(request, "main/lookup.html",
                          context={"session_ids": reversed(session_ids + uploaded_session_ids)})

    else:
        messages.error(request, "You must be an admin to access this feature currently!")
        return redirect("main:homepage")


def upload_session(request):
    if request.user.is_superuser:
        if request.method == "GET":
            return render(request, "main/upload_session.html", context={"form": UploadFileForm()})
        if request.method == "POST":
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                messages.success(request, "Successfully uploaded! Now processing.")
                return redirect("main:session_id_results", session_id=int(form["session_id"].value()), permanent=True)
            messages.error(request, "Form is not valid!")
            return render(request, "main/upload_session.html", context={"form": form})
    else:
        messages.error(request, "You must be an admin to access this feature currently!")
        return redirect("main:homepage")


def register(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            login(request, user)
            messages.success(request, message=f"New account created: {username}")
            return redirect("main:homepage")
        else:
            for msg in form.error_messages:
                print(form.error_messages[msg])
                messages.error(request, f"{msg}: {form.error_messages[msg]}")

    return render(request, "main/register.html", context={"form": NewUserForm})


def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, message=f"Successfully logged into as: {username}")
                return redirect("main:homepage")
            else:
                for msg in form.error_messages:
                    print(form.error_messages[msg])
                messages.error(request, "Invalid username or password")
        else:
            messages.error(request, "Invalid username or password")
    return render(request, "main/login.html", {"form": AuthenticationForm})


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")


def account_page(request):
    if request.user.is_superuser:
        return render(request, "main/admin_options.html")
    else:
        messages.error(request, "You are not a superuser.")
        return redirect("main:homepage")


def run_predictions(request, session_id):
    if request.method == 'GET':
        file_url = request.GET.get('file_url', None)

        if file_url is None:
            warnings.warn(f"Run predictions was called with an invalid file_url: \n {file_url}")

        pred = predict_age(os.path.join(settings.MEDIA_ROOT, file_url.strip(settings.MEDIA_URL)))
        data = {
            'pred': pred
        }
        return JsonResponse(data)


def run_segmentation(request, session_id):
    if request.method == 'GET':
        file_path = request.GET.get('file_url', None)

        if file_path is not None:

            abs_file_path = os.path.join(settings.MEDIA_ROOT, file_path.strip(settings.MEDIA_URL))

            # TODO: Maybe find this puppy a new home
            tmp_path = os.path.join(settings.MEDIA_ROOT, "tmp/")
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            tmp_file_path = brain_segment(brain_path=abs_file_path,
                                          folder_path_to_write=tmp_path,
                                          tmp_file_name=None)

            # Segmented File Path
            tmp_fp = tmp_file_path.split(settings.MEDIA_ROOT.split(os.path.basename(settings.MEDIA_ROOT))[-2][:-1])[-1]

            data = {
                'segmented_file_path': tmp_fp
            }
            return JsonResponse(data)
