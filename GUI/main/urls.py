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
from django.urls import path

from . import views, user_views, result_helper_views

app_name = "main"

urlpatterns = [
    path("", views.homepage, name="homepage"),
    path('lookup/', views.lookup, name="lookup"),
    path('upload/', views.upload_session, name='upload'),
    path('register/', user_views.register, name='register'),
    path('logout/', user_views.logout_request, name='logout'),

    path('login/', user_views.login_request, name='login'),
    path('lookup/login/', user_views.login_request, name='login'),
    path('upload/login/', user_views.login_request, name='login'),
    path('results/login/', user_views.login_request, name='login'),
    path('results/<int:session_id>/login/', user_views.login_request, name='login'),
    path("account/login/", user_views.login_request, name='login'),

    path('results/', views.view_session_results, name="results"),
    path('results/<int:session_id>', views.view_session_results, name="session_id_results"),
    path("results/<int:session_id>/run_predictions/", result_helper_views.run_prediction, name="run_predictions"),
    path("results/<int:session_id>/run_segmentation/", result_helper_views.run_segmentation, name="run_segmentation"),
    path("results/<int:session_id>/remove_tmp/", result_helper_views.remove_tmp, name="remove_tmp"),
    path("load_database/", views.load_data, name="load_database"),
    path("account/", user_views.account_page, name="account"),
    path("about/", views.about, name="about"),
]
