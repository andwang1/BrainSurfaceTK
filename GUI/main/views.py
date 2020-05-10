import os
import json

from django.contrib import messages
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt

from main.custom_wrapper_decorators import custom_login_required, custom_staff_member_required
from main.load_helper import load_original_data
from main.result_helpers import get_mri_js_html, get_surf_file_url, build_session_table, get_unique_session
from .forms import UploadFileForm
from .models import SessionDatabase, UploadedSessionDatabase, Information

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "/main/static/main/data")

SESSIONDATABASES = (SessionDatabase, UploadedSessionDatabase)


def single_slug(request, page_slug):
    results = Information.objects.all().filter(page_slug=page_slug)
    if results.count() == 1:
        page = results.get()
        return render(request=request,
                      template_name='main/blank_template.html',
                      context={"page": page})
    messages.error(request, message="Could not locate webpage.")
    return redirect("main:homepage")


def homepage(request):
    """
    Homepage page
    :return: rendered main/homepage.html with all options available to the user.
    """
    if Information.objects.filter(page_slug="lookup").count() != 1:
        Information.objects.create(page_title="Look-up".title(), page_summary="Look-up session IDs", page_slug="lookup")
    if Information.objects.filter(page_slug="upload").count() != 1:
        Information.objects.create(page_title="Upload".title(), page_summary="Upload session ID", page_slug="upload")
    if Information.objects.filter(page_slug="about").count() != 1:
        Information.objects.create(page_title="About".title(), page_summary="About this project", page_slug="about")
    options = Information.objects.all()
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
        SessionDatabase.objects.all().delete()
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
                    else (int(session.session_id), False) for session in SessionDatabase.objects.all()]

        uploaded_sessions = [(int(session.session_id), True) if session.mri_file != ""
                             else (int(session.session_id), False) for session in UploadedSessionDatabase.objects.all()]
        sessions.extend(uploaded_sessions)
        sessions.sort()
        session_ids, has_mri = zip(*sessions)
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

        pred = round(predict_age(os.path.join(settings.MEDIA_ROOT, file_url.strip(settings.MEDIA_URL))), 3)
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

            current_tmp_files = [f for f in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, f))]
            while True:
                tmp_file_name = randomString(stringLength=10) + ".vtp"
                if tmp_file_name not in current_tmp_files:
                    break

            tmp_file_path = brain_segment(brain_path=abs_file_path,
                                          folder_path_to_write=tmp_path,
                                          tmp_file_name=tmp_file_name)

            # Segmented File Path
            tmp_fp = tmp_file_path.split(settings.MEDIA_ROOT.split(os.path.basename(settings.MEDIA_ROOT))[-2][:-1])[-1]


            data = {
                'segmented_file_path': tmp_fp
            }
            return JsonResponse(data)


# def remove_tmp(request, session_id=None):
#     data = {
#         'success': 'failed'
#     }
#     relative_file_path = request.GET.get('tmp_file_url', None)
#     if relative_file_path is not None:
#         file_path = os.path.join(settings.MEDIA_ROOT, relative_file_path.strip(settings.MEDIA_URL))
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             data['success'] = "success"
#     return JsonResponse(data)


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

def remove_tmp(request, session_id=None):
    if request.method == 'GET':
        if remove_file(request.GET.get('tmp_file_url', None)):
            return JsonResponse({"success": "success"})
        return JsonResponse({"success": "failed"})


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(stringLength))
