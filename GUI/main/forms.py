from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms import ModelForm

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


class LookUpModelChoiceField(forms.ModelChoiceField):

    def label_from_instance(self, obj):
        return f"{obj[0]} : {obj[1]}"


class LookUpForm(forms.Form):
    choices = LookUpModelChoiceField(
        queryset=Session.objects.all().values_list("participant_id", "session_id").order_by("participant_id"),
        label="Select a Participant ID with a Session ID",
        empty_label="Participant ID : Session ID",
        show_hidden_initial=True,
        required=False,
        widget=forms.Select(attrs={"id": "dropdown-session-id", "name": "selected_session_id"}))
