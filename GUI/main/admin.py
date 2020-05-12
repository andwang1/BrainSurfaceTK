from django.contrib import admin
from django.db import models
from tinymce.widgets import TinyMCE

from .models import SessionDatabase, Page

"""
This file contains all the Admin versions of the models. This is used in the admin section of the website to modify 
the users database, options on the home page, modify uploaded sessions & the original data entries.
"""


class PageAdmin(admin.ModelAdmin):
    """
    Used to view/modify options in the home page.
    """

    fieldsets = [
        ("Title/date", {'fields': ["page_title", "page_published"]}),
        ("URL", {'fields': ["page_slug"]}),
        ("Summary", {'fields': ["page_summary"]}),
        ("Content", {"fields": ["page_content"]})
    ]

    formfield_overrides = {
        models.TextField: {'widget': TinyMCE(attrs={'cols': 80, 'rows': 30})},
    }

    search_fields = ('page_title', 'page_published')
    list_display = ('page_title', 'page_published')


class SessionDatabaseAdmin(admin.ModelAdmin):
    """
    Used to view/modify the original session IDs.
    """
    fieldsets = [
        ("Meta Data", {'fields': ["participant_id", "session_id", "gender", "birth_age", "birth_weight", "singleton",
                                  "scan_age", "scan_number", "radiology_score", "sedation"]}),
        ("File Paths", {'fields': ["uploaded", "mri_file", "surface_file"]}),
    ]

    readonly_fields = ("uploaded",)
    search_fields = ('participant_id', 'session_id', 'uploaded')
    list_display = ('session_id', 'participant_id', 'uploaded')


admin.site.register(SessionDatabase, SessionDatabaseAdmin)
admin.site.register(Page, PageAdmin)
