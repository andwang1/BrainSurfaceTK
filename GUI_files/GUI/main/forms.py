from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


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


# class LookUpPatientID(forms.Form):
#     pid = forms.ChoiceField(choices=[(x, x) for x in [0, 1, 2, 6, 10]])  # Temporary for illustrative purposes. . .
#
#     class Meta:
#         fields = ("pid",)
