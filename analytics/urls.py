from django.urls import path
from analytics.views import upload_video,display,runModel,areas
from django.conf.urls.static import static
from django.conf import  settings


urlpatterns = [
    path('upload/',upload_video,name='upload'),
    path('areas/', areas, name='areas'),
    path('videos/<int:area_id>/',display,name='videos'),
    path('runmodels/<int:video>/', runModel),
]