from django.http.request import HttpRequest
from django.http.response import HttpResponse
from analytics.models import CameraArea, Video
from django.shortcuts import get_list_or_404, render,redirect

# Create your views here.
 
def areas(request):
    areas = CameraArea.objects.all()
    context ={
        'areas':areas,
    }
    print(areas)
    return render(request,'index.html',context)


def upload_video(request,area_id):
     
    if request.method == 'POST': 
         
        title = request.POST['title']
        video = request.POST['video']
         
        content = Video(title=title,video=video)
        content.save()
        return redirect('home')
     
    return render(request,'upload.html')

def display(request,area_id):
    videos = get_list_or_404(Video,camera_area = area_id)
    context ={
        'videos':videos,
    }
    print(videos)
    return render(request,'video.html',context)

def runModel(request,video):
    print(video)
    return redirect('/people/analyze/'+str(video))