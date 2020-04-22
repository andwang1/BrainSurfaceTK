from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import UploadedSessionDatabase


class NewUserForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super(NewUserForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
        return user


class UploadFileForm(ModelForm):
    # mri_file = models.FileField(verbose_name="MRI file path", allow_empty_file=True, upload_to="/data/mris/")
    # surface_file = models.FileField(verbose_name="Surface file path", allow_empty_file=False, upload_to="/data/vtps/")

    class Meta:
        model = UploadedSessionDatabase
        fields = ('participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                  'scan_age', 'scan_number', 'radiology_score', 'sedation', 'mri_file', 'surface_file')

    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)
        self.fields['mri_file'].required = False
