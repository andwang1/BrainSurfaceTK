import json
import os

from django.contrib import messages
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from main.custom_wrapper_decorators import custom_login_required, custom_staff_member_required
from main.load_helper import load_original_data
from main.result_helpers import get_mri_js_html, get_surf_file_url, build_session_table, get_unique_session

from .forms import UploadFileForm
from .models import Session, Page

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "/main/static/main/data")

SESSIONDATABASES = (Session,)


def single_slug(request, page_slug):
    results = Page.objects.all().filter(page_slug=page_slug)
    if results.count() == 1:
        page = results.get()
        return render(request=request,
                      template_name=f'main/{page.page_template}',
                      context={"page": page})
    messages.error(request, message="Could not locate webpage.")
    return redirect("main:homepage")


def homepage(request):
    """
    Homepage page
    :return: rendered main/homepage.html with all options available to the user.
    """
    if Page.objects.filter(page_slug="lookup").count() != 1:
        Page.objects.create(page_title="Look-up".title(), page_summary="Look-up session IDs", page_slug="lookup",
                            page_template="lookup.html")
    if Page.objects.filter(page_slug="upload").count() != 1:
        Page.objects.create(page_title="Upload".title(), page_summary="Upload session ID", page_slug="upload",
                            page_template="upload.html")
    if Page.objects.filter(page_slug="about").count() != 1:
        Page.objects.create(page_title="About".title(), page_summary="About this project", page_slug="about",
                            page_template="about.html")
    options = Page.objects.all()
    return render(request, "main/homepage.html", context={"options": options})


@custom_login_required()
def view_session_results(request, session_id=None, display_mri="true"):
    """
    Gets the session ID from the databases and loads the mri file & fetches the vtp file path that is all sent to the
    user to visualise the results. If there is an error with the session_id then the user is redirected to "main:lookup"
    with the associated error message.
    :param session_id: session id that is desired to be visualised.
    :return: rendered main/results.html
    """
    if request.method == "GET":

        record, msg = get_unique_session(session_id, SESSIONDATABASES)

        if record is None:
            messages.error(request, msg)
            return redirect("main:lookup")

        table_names, table_values = build_session_table(record)

        if display_mri.lower() == "true":
            mri_js_html, msg = get_mri_js_html(record)
            if mri_js_html is None and msg is not None:
                messages.error(request, msg)
        else:
            mri_js_html = None

        surf_file_url, msg = get_surf_file_url(record)
        if surf_file_url is None and msg is not None:
            messages.error(request, msg)

        return render(request, "main/results.html",
                      context={"session_id": session_id, "table_names": table_names, "table_values": table_values,
                               "mri_js_html": mri_js_html, "surf_file_url": surf_file_url})


@custom_staff_member_required()
def load_data(request):
    """
    Wipes the SessionDatabase and optionally wipes the UploadedSessionDatabase before loading
     the original dataset into the Django builtin database & detects their associated files
    :return: redirects to homepage with messages to notify the user for success or any errors
    """
    if request.method == "POST":
        # Clear each database here
        Session.objects.all().delete()
        reset_upload_database = request.POST.get("reset_upload_database", 'off')
        if reset_upload_database == 'on':
            reset_upload_database = True

        outcome = load_original_data(reset_upload_database)

        if outcome["success"]:
            messages.success(request, "Successfully loaded data!")
            return redirect("main:homepage")
        else:
            messages.error(request, outcome["message"])

    return render(request, "main/load_database.html")


@custom_login_required()
@csrf_exempt
def lookup(request):
    if request.method == "GET":
        sessions = [(int(session.session_id), True) if session.mri_file != ""
                    else (int(session.session_id), False) for session in Session.objects.all()]
        if len(sessions) > 0:
            session_ids, has_mri = zip(*sessions)
        else:
            session_ids, has_mri = [], []
        return render(request, "main/lookup.html",
                      context={"session_ids": session_ids, "mri_mask": json.dumps(has_mri)})

    if request.method == "POST":
        session_id = request.POST.get("selected_session_id", None)
        if request.POST.get("display-mri", 'off') == 'on':
            display_mri = "true"
        else:
            display_mri = "false"
        if isinstance(session_id, str):
            if session_id.isnumeric():
                return redirect("main:session_id_results", session_id=session_id,
                                display_mri=display_mri, permanent=True)
        messages.warning(request, message="Please select a session ID")
        return redirect("main:lookup", permanent=True)


@custom_login_required(login_url="login/")
def upload_session(request):
    if request.method == "GET":
        return render(request, "main/upload_session.html", context={"form": UploadFileForm()})
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Successfully uploaded! Now processing.")
            return redirect("main:session_id_results", session_id=int(form["session_id"].value()),
                            display_mri="true", permanent=True)
        messages.error(request, "Form is not valid!")
        return render(request, "main/upload_session.html", context={"form": form})
