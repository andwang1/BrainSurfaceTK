from datetime import datetime
from django.core import validators
from django.db import models
"""
THESE ARE YOUR DATABASES BRO
"""
class Page(models.Model):
    page_title = models.CharField(max_length=200, unique=True)
    page_summary = models.CharField(max_length=200, blank=True)
    page_content = models.TextField(blank=True)
    page_published = models.DateTimeField('date published', default=datetime.now, blank=True)
    page_slug = models.SlugField(max_length=100, unique=True, blank=True)
    def __str__(self):
        return self.page_title
def get_upload_path(instance, filename):
    out = ""
    if instance.uploaded:
        out += "uploads/data/"
    else:
        out += "original/data/"
    if filename.endswith(".nii"):
        return f"{out}/mris/{filename}"
    else:
        return f"{out}/vtps/{filename}"
class Session(models.Model):
    """
    General form for session records to be inserted into
    """
    participant_id = models.CharField(verbose_name="Participant ID", max_length=100, blank=True)
    session_id = models.IntegerField(verbose_name="Session ID", validators=[validators.MinValueValidator(0)])
    gender = models.CharField(verbose_name="Gender", max_length=100, blank=True)
    birth_age = models.FloatField(verbose_name="Birth Age", blank=True)
    birth_weight = models.FloatField(verbose_name="Birth Weight", blank=True)
    singleton = models.CharField(verbose_name="Singleton", max_length=100, blank=True)
    scan_age = models.FloatField(verbose_name="Scan Age", blank=True)
    scan_number = models.IntegerField(verbose_name="Scan Number", blank=True)
    radiology_score = models.CharField(verbose_name="Radiology Score", max_length=200, blank=True)
    sedation = models.CharField(verbose_name="Sedation", max_length=200, blank=True)
    uploaded = models.BooleanField(verbose_name="Uploaded", default=True)
    mri_file = models.FileField(verbose_name="MRI file path", upload_to=get_upload_path, max_length=250, blank=True,
                                validators=[validators.FileExtensionValidator(allowed_extensions=["", "nii"])])
    surface_file = models.FileField(verbose_name="Surface file path", upload_to=get_upload_path, max_length=250,
                                    blank=True,
                                    validators=[validators.FileExtensionValidator(allowed_extensions=["", "vtp"])])
    class Meta:
        ordering = ['-session_id']
        unique_together = ('participant_id', 'session_id',)
        verbose_name_plural = "Sessions"
    def __str__(self):
        return f"Session ID: {self.session_id}"