from django.contrib import admin
from tinymce.widgets import TinyMCE
from django.db import models
from .models import Option, SessionDatabase, UploadedSessionDatabase


# Register your models here.
class OptionAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Option", {'fields': ["option"]}),
        ("URL", {'fields': ["url"]}),
        ("Description", {'fields': ["description"]}),
    ]

    formfield_overrides = {
        models.TextField: {'widget': TinyMCE(attrs={'cols': 80, 'rows': 30})},
    }


class PatientEntryAdmin(admin.ModelAdmin):
    fieldsets = [
        ("ID", {'fields': ["id"]}),
        ("URL", {'fields': ["url"]}),
    ]


class UploadedSessionDatabaseAdmin(admin.ModelAdmin):
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