from django.urls import path
from analytics.views import upload_video,display,runAnalytic
 
from django.conf.urls.static import static
from django.conf import  settings
 
 
 
urlpatterns = [
    path('upload/',upload_video,name='upload'),
    path('',display,name='videos'),
    path('runmodels/', runAnalytic),
]