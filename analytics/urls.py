from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/<int:video_id>/', views.runAnalytic, name='runAnalytic'),
]