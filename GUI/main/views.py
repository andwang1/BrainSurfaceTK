from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import permission_required
from django.contrib import messages
from .models import Option, SessionDatabase, UploadedSessionDatabase
from .forms import NewUserForm, UploadFileForm
from nilearn.plotting import view_img
import nibabel as nib
import os
import sys
from django.http import JsonResponse
import csv
from django.views.decorators.csrf import csrf_exempt

# from .evaluate_pointnet_regression import predict_age

BASE_DIR = os.getcwd()
DATA_DIR = f"{BASE_DIR}/main/static/main/data"
VOL_DIR = f"{DATA_DIR}/gm_volume3d"
SURF_DIR = f"{DATA_DIR}/vtp"


def homepage(request):
    if Option.objects.count() == 0:
        Option.objects.create(name="Look-up", summary="Look-up session ids", slug="lookup")
        Option.objects.create(name="Upload", summary="Upload session id", slug="upload")
        Option.objects.create(name="About", summary="About this project", slug="about")
    options = Option.objects.all()
    return render(request, "main/homepage.html", context={"options": options})


def about(request):
    return render(request, "main/about.html")


def view_session_results(request, session_id=None):
    # NEW
    # TODO: Implement method to only run prediction model when button clicked and update currently rendered page
    if request.method == "GET":
        session_id = request.GET["session_id"]
        database = SessionDatabase
    if request.method == "POST":
        database = UploadedSessionDatabase

    search_results = database.objects.filter(session_id=session_id)
    if search_results.count() is 1:
        record = search_results.get()
        record_dict = vars(record)
        # TODO: Rewrite to exclude clutter
        field_names = record_dict.keys()
        table_names = list()
        table_values = list()
        for field_name in field_names:
            if field_name.startswith("_") or field_name == "id" or field_name.endswith("file"):
                continue
            table_names.append(field_name.replace("_", " ").lower().capitalize())
            table_values.append(record_dict[field_name])

        mri_file = record.mri_file
        mri_js_html = None
        # TODO: add check that file ends with .nii
        if mri_file.name != "":
            if os.path.isfile(mri_file.path) & mri_file.path.endswith("nii"):
                img = nib.load(mri_file.path)
                mri_js_html = view_img(img, colorbar=False, bg_img=False, black_bg=True, cmap='gray')
            else:
                messages.error(request, "ERROR: Either MRI file doesn't exist or doesn't end with .nii!")

        surf_file = record.surface_file
        surf_file_path = None
        # TODO: add check that file ends with .vtp
        if surf_file.name != "":
            surf_file_path = surf_file.path.split("media/")[-1]
            if not (os.path.isfile(surf_file.path) & surf_file.path.endswith("vtp")):
                surf_file_path = None
                messages.error(request, "ERROR: Either Surface file doesn't exist or doesn't end with .vtp!")

        # TODO: Create model for storing vtk file locations that can be looked up properly
        pred = None
        # predict_age("{SURF_DIR}/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp")

        return render(request, "main/results.html",
                      context={"session_id": session_id, "table_names": table_names, "table_values": table_values,
                               "mri_js_html": mri_js_html, "surf_file_path": surf_file_path, "pred": pred})
    else:
        messages.error(request, "ERROR: Multiple records were found, session id is not unique!")
        return redirect("main:lookup")


@permission_required('admin.can_add_log_entry')
def load_data(request):
    if request.method == "POST":
        # Clear each database here
        SessionDatabase.objects.all().delete()

        # Check if the tsv file exists
        if not os.path.isfile(SessionDatabase.default_tsv_path):
            messages.error(request, "Either this is not a file or the location is wrong!")
            return redirect("main:load_database")

        expected_ordering = ['participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                             'scan_age', 'scan_number', 'radiology_score', 'sedation']

        # [f for f in os.listdir(SessionDatabase.default_mris_path) if f.endswith("nii")]
        found_mri_files = [f for f in os.listdir(SessionDatabase.default_mris_path) if
                           f.endswith("nii")]
        # TODO: Check what key words will be in files we want!
        found_vtps_files = [f for f in os.listdir(SessionDatabase.default_vtps_path) if
                            f.endswith("vtp") & f.find("inflated") != -1 & f.find("whole")]

        with open(SessionDatabase.default_tsv_path) as foo:
            reader = csv.reader(foo, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    if row != expected_ordering:
                        messages.error(request, "FAILED! The table did not have the expected column name ordering.")
                        return redirect("main:load_database")
                    continue

                (participant_id, session_id, gender, birth_age, birth_weight, singleton, scan_age,
                 scan_number, radiology_score, sedation) = row

                mri_file = next((f"{SessionDatabase.default_mris_path}/{x}" for x in found_mri_files if
                                 (participant_id and session_id) in x), "")
                surface_file = next((f"{SessionDatabase.default_vtps_path}/{x}" for x in found_vtps_files if
                                     (participant_id and session_id) in x), "")

                try:
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
                                                   mri_file=mri_file,
                                                   surface_file=surface_file)
                except:
                    # TODO: Investigate these errors, why so many duplicates???
                    # messages.error(request, f'Warning: tsv contains non-uniques session id: {session_id}')
                    print(f'tsv contains non-uniques session id: {session_id}')

        messages.success(request, "Successfully loaded data!")
        return redirect("main:homepage")

    return render(request, "main/load_database.html")


def lookup(request):
    if request.user.is_superuser:
        if request.method == "GET":
            session_ids = sorted([int(session.session_id) for session in SessionDatabase.objects.all()])
            return render(request, "main/lookup.html", context={"session_ids": session_ids})
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
                return view_session_results(request, session_id=int(form["session_id"].value()))
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

    form = NewUserForm
    return render(request,
                  "main/register.html",
                  context={"form": form})


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
    form = AuthenticationForm()
    return render(request,
                  "main/login.html",
                  {"form": form})


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


def lookup(request):
    if request.method == "GET":
        session_ids = sorted([int(session.session_id) for session in GreyMatterVolume.objects.all()])
        return render(request, "main/lookup.html", context={"session_ids": session_ids})

@csrf_exempt
def run_predictions(request):

    if request.method == 'POST':

        # pred = predict_age(SURF_DIR + "/sub-CC00050XX01_ses-7201_hemi-L_inflated_reduce50.vtp")
        pred = 42

        data = {
            'pred': pred
        }
        return JsonResponse(data)
