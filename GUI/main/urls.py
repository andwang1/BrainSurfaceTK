"""BasicSite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.views.generic import TemplateView
from django.conf.urls.static import static
from django.conf import settings

app_name = "main"

urlpatterns = [
    path("", views.homepage, name="homepage"),
    path('lookup/', views.lookup, name="lookup"),
    path('upload/', views.upload_session, name='upload'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_request, name='logout'),
    path('login/', views.login_request, name='login'),
    path('results/', views.view_session_results, name="results"),
    path("load_database/", views.load_data, name="load_database"),
    path("account/", views.account_page, name="account"),
    path("about/", views.about, name="about"),
    path("run_predictions/", views.run_predictions, name="run_predictions")
    # path("brain_surf/", TemplateView.as_view(template_name="main/index.html"),
    #                    name='brain_surf'),
]  # + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
