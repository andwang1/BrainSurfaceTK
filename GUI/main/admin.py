from django.contrib import admin
from django.db import models
from tinymce.widgets import TinyMCE

from .models import Option, SessionDatabase, UploadedSessionDatabase

"""
This file contains all the Admin versions of the models. This is used in the admin section of the website to modify 
the users database, options on the home page, modify uploaded sessions & the original data entries.
"""


# Register your models here.
class OptionAdmin(admin.ModelAdmin):
    """
    Used to view/modify options in the home page.
    """

    fieldsets = [
        ("Option", {'fields': ["option"]}),
        ("URL", {'fields': ["url"]}),
        ("Description", {'fields': ["description"]}),
    ]

    formfield_overrides = {
        models.TextField: {'widget': TinyMCE(attrs={'cols': 80, 'rows': 30})},
    }


class UploadedSessionDatabaseAdmin(admin.ModelAdmin):
    """
    Used to view/modify uploaded sessions IDs.
    """
    fieldsets = [
        ("Participant_id", {'fields': ["participant_id"]}),
        ("session_id", {'fields': ["session_id"]}),
        ("gender", {'fields': ["gender"]}),
        ("birth_age", {'fields': ["birth_age"]}),
        ("birth_weight", {'fields': ["birth_weight"]}),
        ("singleton", {'fields': ["singleton"]}),
        ("scan_age", {'fields': ["scan_age"]}),
        ("scan_number", {'fields': ["scan_number"]}),
        ("radiology_score", {'fields': ["radiology_score"]}),
        ("sedation", {'fields': ["sedation"]}),
        ("mri_file", {'fields': ["mri_file"]}),
        ("surface_file", {'fields': ["surface_file"]}),
    ]


class SessionDatabaseAdmin(admin.ModelAdmin):
    """
    Used to view/modify the original session IDs.
    """
    fieldsets = [
        ("Participant_id", {'fields': ["participant_id"]}),
        ("session_id", {'fields': ["session_id"]}),
        ("gender", {'fields': ["gender"]}),
        ("birth_age", {'fields': ["birth_age"]}),
        ("birth_weight", {'fields': ["birth_weight"]}),
        ("singleton", {'fields': ["singleton"]}),
        ("scan_age", {'fields': ["scan_age"]}),
        ("scan_number", {'fields': ["scan_number"]}),
        ("radiology_score", {'fields': ["radiology_score"]}),
        ("sedation", {'fields': ["sedation"]}),
    ]


admin.site.register(Option)
admin.site.register(SessionDatabase)
admin.site.register(UploadedSessionDatabase)
