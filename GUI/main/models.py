from django.db import models
import os

"""
THESE ARE YOUR DATABASES BRO
"""

BASE_DIR = os.getcwd()
MEDIA_DATA_DIR = f"{BASE_DIR}/media/original/data"


# Create your models here.
class Option(models.Model):
    name = models.CharField(max_length=200)
    summary = models.CharField(max_length=200)
    slug = models.CharField(max_length=200)

    def __str__(self):
        return self.name


class TemplateSessionDatabase(models.Model):
    participant_id = models.CharField(verbose_name="participant_id", max_length=100)
    session_id = models.IntegerField(verbose_name="session_id", unique=True, primary_key=True)
    gender = models.CharField(verbose_name="gender", max_length=100)
    birth_age = models.FloatField(verbose_name="birth_age")
    birth_weight = models.FloatField(verbose_name="birth_weight")
    singleton = models.CharField(verbose_name="singleton", max_length=100)
    scan_age = models.FloatField(verbose_name="scan_age")
    scan_number = models.IntegerField(verbose_name="scan_number")
    radiology_score = models.CharField(verbose_name="radiology_score", max_length=200)
    sedation = models.CharField(verbose_name="sedation", max_length=200)

    mri_file = models.FileField(verbose_name="MRI file path", upload_to="uploads/data/mris/", default="")
    surface_file = models.FileField(verbose_name="Surface file path", upload_to="uploads/data/vtps/", default="")

    class Meta:
        abstract = True

    def __str__(self):
        return f"Session ID: {self.session_id}"


class UploadedSessionDatabase(TemplateSessionDatabase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mri_file.upload_to = "original/data/mris/"
        self.surface_file.upload_to = "original/data/vtps/"

    # mri_file = models.FileField(verbose_name="MRI file path", upload_to="uploads/data/mris/", default="")
    # surface_file = models.FileField(verbose_name="Surface file path", upload_to="uploads/data/vtps/", default="")

    class Meta:
        ordering = ['-session_id']
        verbose_name_plural = "Uploaded Session Database"


class SessionDatabase(TemplateSessionDatabase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mri_file.upload_to = "original/data/mris/"
        self.surface_file.upload_to = "original/data/vtps/"

    # mri_file = models.FileField(verbose_name="MRI file path", upload_to="original/data/mris/", default="")
    # surface_file = models.FileField(verbose_name="Surface file path", upload_to="original/data/vtps/", default="")

    class Meta:
        ordering = ['-session_id']
        verbose_name_plural = "Session Database"

    default_tsv_path = f"{MEDIA_DATA_DIR}/meta_data.tsv"
    default_vtps_path = f"{MEDIA_DATA_DIR}/vtps"
    default_mris_path = f"{MEDIA_DATA_DIR}/mris"
