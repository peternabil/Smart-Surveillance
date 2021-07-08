from MainApp.models import Video
from django.db import models

# Create your models here.
class Analytic(models.Model):
    video = models.ForeignKey(to=Video,on_delete=models.CASCADE)
    avg_persons_num = models.BigIntegerField()
    avg_male = models.BigIntegerField()
    avg_female = models.BigIntegerField()
    def create_analytic(self, video):
        analytic = self.create(video = video)
        return analytic

    def __str__(self):
        return self.video.name

    class Meta:
        ordering = ['avg_persons_num']
