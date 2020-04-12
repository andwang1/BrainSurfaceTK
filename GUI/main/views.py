from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.template import loader
from .models import Option, SessionDatabase, GreyMatterVolume
from .forms import NewUserForm
from nilearn.plotting import view_img
import nibabel as nib
import os
import csv

DATA_DIR = "./main/static/main/data"
VOL_DIR = f"{DATA_DIR}/gm_volume3d"

# Different because this is used in HTML with the static tag!
SURF_DIR = f"main/data/vtp"


# Create your views here.
def view_surf(request):
    'sub-CC00050XX01_ses-7201_left_pial'
    "filePath/forCem.vtp"
    # render(request, "main/brain_surf.html", context={"options": None,
    #                                                  "fileURL": f"{SURF_DIR}/forCem.vtp"})
    return loader.render_to_string("main/index.html", context={"fileURL": f"{SURF_DIR}/forCem.vtp"}, request=request)

def homepage(request):
    if Option.objects.count() == 0:
        Option.objects.create(name="Look-up", summary="Look-up session ids", slug="lookup")
        Option.objects.create(name="Upload", summary="Upload session id", slug="upload")
    options = Option.objects.all()
    return render(request, "main/homepage.html", context={"options": options})


# def view_session_data(request, session_id):
#     record = SessionDatabase.objects.filter(session_id=session_id)
#     information = record.values()
#     column_names = information.query.values_select[1:]  # Drop sql id col
#     values = list(*record.values_list())[1:]
#     file_path = list(*GreyMatterVolume.objects.filter(session_id=session_id).values_list())[-1]
#     img = nib.load(file_path)
#     img_html = view_img(img, colorbar=False, bg_img=False, cmap='gray')
#     participant_id = values[0]
#     file_name = f"sub-{participant_id}_ses-{session_id}_left_pial.vtp"
#     return render(request, "main/results.html",
#                   context={"session_id": session_id, "column_names": column_names, "values": values,
#                            "image": img_html, "fileURL": f"{SURF_DIR}/forCem.vtp"})

def view_session_data(request, session_id):
    record = SessionDatabase.objects.filter(session_id=session_id)
    information = record.values()
    column_names = information.query.values_select[1:]  # Drop sql id col
    values = list(*record.values_list())[1:]
    file_path = list(*GreyMatterVolume.objects.filter(session_id=session_id).values_list())[-1]
    img = nib.load(file_path)
    img_html = view_img(img, colorbar=False, bg_img=False, cmap='gray')
    participant_id = values[0]
    file_name = f"sub-{participant_id}_ses-{session_id}_left_pial.vtp"
    return render(request, "main/results.html",
                  context={"session_id": session_id, "column_names": column_names, "values": values,
                           "image_slicer_window": img_html, "fileURL": f"{SURF_DIR}/forCem.vtp"})



def load_data(request):
    if request.method == "POST":
        SessionDatabase.objects.all().delete()
        GreyMatterVolume.objects.all().delete()
        with open(f"{DATA_DIR}/meta_data.tsv") as foo:
            reader = csv.reader(foo, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                (participant_id, session_id, gender, birth_age, birth_weight, singleton, scan_age,
                 scan_number, radiology_score, sedation) = row
                try:
                    SessionDatabase.objects.get_or_create(participant_id=participant_id, session_id=session_id,
                                                          gender=gender,
                                                          birth_age=birth_age, birth_weight=birth_weight,
                                                          singleton=singleton,
                                                          scan_age=scan_age,
                                                          scan_number=scan_number, radiology_score=radiology_score,
                                                          sedation=sedation)
                except:
                    # TODO: This is a temporary patch because we have two session ids of 1000
                    print('tmp')
        pids_and_session_ids_zip = sorted(
            [(session.participant_id, session.session_id) for session in SessionDatabase.objects.all()])
        potential_files = []
        for file in os.listdir(VOL_DIR):
            if file.endswith(".nii"):
                potential_files.append(file)

        for pid, session_id in pids_and_session_ids_zip:
            for file in potential_files:
                if file.rfind(session_id, 10, 30) > -1 and file.rfind(pid, 0, 15) > -1:
                    try:
                        GreyMatterVolume.objects.get_or_create(session_id=session_id,
                                                               participant_id=participant_id,
                                                               filename=file, path=VOL_DIR,
                                                               filepath=f"{VOL_DIR}/{file}")
                        potential_files.remove(file)
                    except:
                        # TODO: This is a temporary patch because we have two session ids of 1000
                        print("tmp")
                    break

        messages.success(request, "Successfully loaded data!")
        return redirect("main:homepage")
    return render(request, "main/load_database.html")


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


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")


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


def account_page(request):
    if request.user.is_superuser:
        return redirect("../admin")
    else:
        messages.error(request, "You are not a superuser.")
        return redirect("main:homepage")


def lookup(request):
    if request.method == "POST":
        session_id = request.POST['selected_id']
        return view_session_data(request, session_id)
    session_ids = sorted([int(session.session_id) for session in GreyMatterVolume.objects.all()])
    return render(request, "main/lookup.html", context={"session_ids": session_ids})
