
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    url(r'^$', views.home),
]

urlpatterns += staticfiles_urlpatterns()