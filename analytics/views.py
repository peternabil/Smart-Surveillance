from django.http.request import HttpRequest
from django.http.response import HttpResponse
from analytics.models import Video
from django.shortcuts import render,redirect

# Create your views here.
def upload_video(request):
     
    if request.method == 'POST': 
         
        title = request.POST['title']
        video = request.POST['video']
         
        content = Video(title=title,video=video)
        content.save()
        return redirect('home')
     
    return render(request,'upload.html')
 
 
def display(request):
     
    videos = Video.objects.all()
    context ={
        'videos':videos,
    }
    print(videos)
    return render(request,'video.html',context)

def runAnalytic(request):
    print(request.GET['video'])
    return HttpResponse(request.GET['video'])