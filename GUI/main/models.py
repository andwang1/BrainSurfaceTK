from datetime import datetime

from django.core import validators
from django.db import models

import os

"""
THESE ARE YOUR DATABASES BRO
"""


class Page(models.Model):
    page_title = models.CharField(max_length=200, unique=True)
    page_summary = models.CharField(max_length=200, blank=True)
    page_content = models.TextField(blank=True)
    page_published = models.DateTimeField('date published', default=datetime.now, blank=True)
    page_template = models.CharField(verbose_name="Page Template", default="blank_template.html", max_length=100)
    page_slug = models.SlugField(max_length=100, unique=True, blank=True)

    def __str__(self):
        return self.page_title


def get_upload_path(instance, filename):
    out = ""
    if instance.uploaded:
        out = os.path.join(out, "uploads", "data")
    else:
        out = os.path.join(out, "original", "data")
    if filename.endswith(".nii"):
        return os.path.join(out, "mris", filename)
    else:
        return os.path.join(out, "vtps", filename)


class Session(models.Model):
    """
    General form for session records to be inserted into
    """
    participant_id = models.CharField(verbose_name="Participant ID", max_length=100)
    session_id = models.IntegerField(verbose_name="Session ID", validators=[validators.MinValueValidator(0)])
    gender = models.CharField(verbose_name="Gender", max_length=100, blank=True, null=True)
    birth_age = models.FloatField(verbose_name="Birth Age", blank=True, null=True)
    birth_weight = models.FloatField(verbose_name="Birth Weight", blank=True, null=True)
    singleton = models.CharField(verbose_name="Singleton", max_length=100, blank=True, null=True)
    scan_age = models.FloatField(verbose_name="Scan Age", blank=True, null=True)
    scan_number = models.IntegerField(verbose_name="Scan Number", blank=True, null=True)
    radiology_score = models.CharField(verbose_name="Radiology Score", max_length=200, blank=True, null=True)
    sedation = models.CharField(verbose_name="Sedation", max_length=200, blank=True, null=True)
    uploaded = models.BooleanField(verbose_name="Uploaded", default=True, null=True)
    mri_file = models.FileField(verbose_name="MRI file path", upload_to=get_upload_path, max_length=250, blank=True,
                                validators=[validators.FileExtensionValidator(allowed_extensions=["", "nii", "nii.gz"])], null=True)
    surface_file = models.FileField(verbose_name="Surface file path", upload_to=get_upload_path, max_length=250,
                                    blank=True, null=True,
                                    validators=[validators.FileExtensionValidator(allowed_extensions=["", "vtp"])])

    class Meta:
        ordering = ['participant_id', 'session_id']
        unique_together = ('participant_id', 'session_id',)
        verbose_name_plural = "Sessions"

    def __str__(self):
        return f"Session ID: {self.session_id}"
