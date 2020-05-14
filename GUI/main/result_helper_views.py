from django.http import JsonResponse
from main.result_helpers import remove_file, pointnet_run_segmentation, pointnet_run_prediction


def run_prediction(request, **kwargs):
    if request.method == 'GET':
        file_url = request.GET.get('file_url', None)
        if file_url is not None:
            return JsonResponse({'pred': pointnet_run_prediction(file_url)})


def run_segmentation(request, **kwargs):
    if request.method == 'GET':
        file_url = request.GET.get('file_url', None)
        if file_url is not None:
            return JsonResponse({'segmented_file_path': pointnet_run_segmentation(file_url)})


def remove_tmp(request, **kwargs):
    if request.method == 'GET':
        if remove_file(request.GET.get('tmp_file_url', None)):
            return JsonResponse({"success": "success"})
        return JsonResponse({"success": "failed"})
