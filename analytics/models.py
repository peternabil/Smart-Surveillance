from django.db import models

# Create your models here.
class CameraArea(models.Model):
    id = models.UUIDField()
    name = models.CharField(max_length=100)
    capacity = models.BigIntegerField()
    def __str__(self):
        return self.name
    class Meta:
        ordering = ['capacity']


class Video(models.Model):
    name = models.CharField(max_length=100)
    video = models.FileField(upload_to='videos/')
    datetime = models.DateTimeField()
    camera_area = models.ForeignKey(to=CameraArea,on_delete=models.CASCADE)

    class Meta:
        verbose_name = 'video'
        verbose_name_plural = 'videos'
    def __str__(self):
        return self.name


