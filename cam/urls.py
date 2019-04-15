from django.urls import path
from . import views
app_name ='cam'
urlpatterns = [
    path('', views.index, name = 'index'),
    path('camera/', views.camera, name = 'camera'),
    path('upload/', views.upload, name = 'upload'),
]
