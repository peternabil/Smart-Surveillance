from people_counter.models import Analytic
from analytics.models import CameraArea, Video
from django.contrib import admin

# Register your models here.
admin.site.register(Video)
admin.site.register(CameraArea)
admin.site.register(Analytic)
