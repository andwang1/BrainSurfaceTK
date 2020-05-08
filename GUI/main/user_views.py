from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

from .forms import NewUserForm


def account_page(request):
    if request.user.is_superuser:
        return render(request, "main/admin_options.html")
    else:
        messages.error(request, "You are not a superuser.")
        return redirect("main:homepage")


def register(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            login(request, user)
            messages.success(request, message=f"New account created: {username}")
            return redirect("main:homepage")
        else:
            for msg in form.error_messages:
                print(form.error_messages[msg])
                messages.error(request, f"{msg}: {form.error_messages[msg]}")

    return render(request, "main/register.html", context={"form": NewUserForm})


def login_request(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, message=f"Successfully logged into as: {username}")
                return redirect("main:homepage")
            else:
                for msg in form.error_messages:
                    print(form.error_messages[msg])
                messages.error(request, "Invalid username or password")
        else:
            messages.error(request, "Invalid username or password")
    return render(request, "main/login.html", {"form": AuthenticationForm})


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("main:homepage")
