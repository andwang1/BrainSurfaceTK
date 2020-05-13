from django import forms
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import Session


class NewUserForm(UserCreationForm):
    """
    The Registration sign-up form used to create a new account.
    """
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
    """
    The form used to upload a new session with the .vtp & mri files attached.
    """
    class Meta:
        model = Session
        fields = ('participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                  'scan_age', 'scan_number', 'radiology_score', 'sedation', 'mri_file', 'surface_file')

    def __init__(self, *args, **kwargs):
        super(UploadFileForm, self).__init__(*args, **kwargs)

