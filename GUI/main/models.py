import os
from datetime import datetime

from django.conf import settings
from django.core import validators
from django.core.exceptions import ValidationError
from django.db import models

"""
THESE ARE YOUR DATABASES BRO
"""


class Page(models.Model):
    page_title = models.CharField(max_length=200, unique=True)
    page_summary = models.CharField(max_length=200)
    page_content = models.TextField()
    page_published = models.DateTimeField('date published', default=datetime.now)
    page_slug = models.CharField(max_length=200, default=1, unique=True)

    def __str__(self):
        return self.page_title


def validate_session_id_is_unique(session_id):
    """
    Checks that this session ID is not already in the database.
    :param session_id: integer value that is checked for uniqueness
    :return: None if no errors, else raises a Validation Error if session id is non-unique
    """
    if (SessionDatabase.objects.all().filter(session_id=session_id).count() > 0) or \
            (UploadedSessionDatabase.objects.all().filter(session_id=session_id).count() > 0):
        raise ValidationError(f'{session_id} is already in the database!', params={'session_id': session_id})


class TemplateSessionDatabase(models.Model):
    """
    General form for session records to be inserted into
    """
    participant_id = models.CharField(verbose_name="Participant ID", max_length=100)
    session_id = models.IntegerField(verbose_name="Session ID", unique=True, primary_key=True,
                                     validators=[validate_session_id_is_unique, validators.MinValueValidator(0)])
    gender = models.CharField(verbose_name="Gender", max_length=100)
    birth_age = models.FloatField(verbose_name="Birth Age")
    birth_weight = models.FloatField(verbose_name="Birth Weight")
    singleton = models.CharField(verbose_name="Singleton", max_length=100)
    scan_age = models.FloatField(verbose_name="Scan Age")
    scan_number = models.IntegerField(verbose_name="Scan Number")
    radiology_score = models.CharField(verbose_name="Radiology Score", max_length=200)
    sedation = models.CharField(verbose_name="Sedation", max_length=200)

    mri_file = models.FileField(verbose_name="MRI file path", upload_to="", default="", max_length=250)
    surface_file = models.FileField(verbose_name="Surface file path", upload_to="", default="", max_length=250)

    class Meta:
        abstract = True

    def __str__(self):
        return f"Session ID: {self.session_id}"


class UploadedSessionDatabase(TemplateSessionDatabase):
    """
    Inheriting from the TemplateSessionDatabase, the main modifications here is the additional file fields that
    accept vtps & nii which are used to render the brain. This class also contains where these file
    types would be stored.
    """
    mri_file_storage_path = "uploads/data/mris/"
    surface_file_storage_path = "uploads/data/vtps/"

    mri_file = models.FileField(verbose_name="MRI file path", upload_to=mri_file_storage_path, default="",
                                max_length=250)
    surface_file = models.FileField(verbose_name="Surface file path", upload_to=surface_file_storage_path, default="",
                                    max_length=250)

    class Meta:
        ordering = ['-session_id']
        verbose_name_plural = "Uploaded Session Database"


class SessionDatabase(TemplateSessionDatabase):
    """
    Inheriting from the TemplateSessionDatabase, the main modifications here is the additional file fields that
    accept vtps & nii which are used to render the brain. This class also contains where these file
    types would be stored.
    """

    mri_file_storage_path = "original/data/mris/"
    surface_file_storage_path = "original/data/vtps/"

    mri_file = models.FileField(verbose_name="MRI file path", upload_to=mri_file_storage_path, default="",
                                max_length=250)
    surface_file = models.FileField(verbose_name="Surface file path", upload_to=surface_file_storage_path, default="",
                                    max_length=250)

    tsv_path = os.path.join(settings.MEDIA_ROOT, "original/data/meta_data.tsv")
    default_mri_path = os.path.join(settings.MEDIA_ROOT, mri_file_storage_path)
    default_vtps_path = os.path.join(settings.MEDIA_ROOT, surface_file_storage_path)

    class Meta:
        ordering = ['-session_id']
        verbose_name_plural = "Session Database"
