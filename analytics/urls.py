from django.urls import path
from video_content.views import upload_video,display
 
from django.conf.urls.static import static
from django.conf import  settings
 
 
 
urlpatterns = [
    path('upload/',upload_video,name='upload'),
    path('videos/',display,name='videos'),
]
 
     
urlpatterns  += static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)