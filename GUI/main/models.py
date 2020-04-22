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


class UploadedSessionDatabase(models.Model):
    participant_id = models.CharField(verbose_name="participant_id", max_length=100)
    session_id = models.IntegerField(verbose_name="session_id", unique=True)
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
        # Gives the proper plural name for admin
        order_with_respect_to = 'session_id'
        verbose_name_plural = "Uploaded Session IDs"

    def __str__(self):
        return f"Session-{self.session_id}"


class SessionDatabase(models.Model):
    participant_id = models.CharField(verbose_name="participant_id", max_length=100)
    session_id = models.IntegerField(verbose_name="session_id", unique=True)
    gender = models.CharField(verbose_name="gender", max_length=100)
    birth_age = models.FloatField(verbose_name="birth_age")
    birth_weight = models.FloatField(verbose_name="birth_weight")
    singleton = models.CharField(verbose_name="singleton", max_length=100)
    scan_age = models.FloatField(verbose_name="scan_age")
    scan_number = models.IntegerField(verbose_name="scan_number")
    radiology_score = models.CharField(verbose_name="radiology_score", max_length=200)
    sedation = models.CharField(verbose_name="sedation", max_length=200)

    mri_file = models.FileField(verbose_name="MRI file path", upload_to="original/data/mris/", default="")
    surface_file = models.FileField(verbose_name="Surface file path", upload_to="original/data/vtps/", default="")

    class Meta:
        # order_with_respect_to = 'session_id'
        ordering = ['-session_id']
        # Gives the proper plural name for admin
        verbose_name_plural = "Session Database"

    default_tsv_path = f"{MEDIA_DATA_DIR}/meta_data.tsv"
    default_vtps_path = f"{MEDIA_DATA_DIR}/vtps"
    default_mris_path = f"{MEDIA_DATA_DIR}/mris"

    def __str__(self):
        return f"Session ID: {self.session_id}"

# class SessionDatabase(models.Model):
#     participant_id = models.CharField(verbose_name="participant_id", max_length=100)
#     session_id = models.CharField(verbose_name="session_id", max_length=100, unique=True)
#     gender = models.CharField(verbose_name="gender", max_length=100)
#     birth_age = models.CharField(verbose_name="birth_age", max_length=100)
#     birth_weight = models.CharField(verbose_name="birth_weight", max_length=100)
#     singleton = models.CharField(verbose_name="singleton", max_length=100)
#     scan_age = models.CharField(verbose_name="scan_age", max_length=50)
#     scan_number = models.CharField(verbose_name="scan_number", max_length=50)
#     radiology_score = models.CharField(verbose_name="radiology_score", max_length=200)
#     sedation = models.CharField(verbose_name="sedation", max_length=200)
#
#     class Meta:
#         # Gives the proper plural name for admin
#         verbose_name_plural = "Session Database"
#
#     def __str__(self):
#         return f"Session ID: {self.session_id}"
#
#
# class GreyMatterVolume(models.Model):
#     session_id = models.CharField(verbose_name="session_id", max_length=100, unique=True)
#     participant_id = models.CharField(verbose_name="participant_id", max_length=100)
#     filename = models.CharField(verbose_name="filename", max_length=200)
#     path = models.CharField(verbose_name="path", max_length=200)
#     filepath = models.CharField(verbose_name="filepath", max_length=200)
#
#     def __str__(self):
#         return f"Session ID: {self.session_id}"
