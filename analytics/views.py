from MainApp.models import Video
from django.http.response import HttpResponse
from django.shortcuts import render, get_object_or_404
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont
import base64
import cv2
from Analytics import mainDriver
import numpy as np
import tensorflow as tf
from Analytics.object_detection.utils import load_class_names, output_boxes, draw_outputs, resize_image
from Analytics.object_detection.yolov3 import YOLOv3Net
from pathlib import Path
import dlib
import os
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from Analytics.gender_age.src.factory import get_model
from Analytics.gender_age.get_gender_age import main_driver

# global settings.
model_size = (416, 416,3)
num_classes = 80

class_name = 'Analytics/object_detection/data/coco.names'
max_output_size = 40
max_output_size_per_class= 20

iou_threshold = 0.5

confidence_threshold = 0.5

cfgfile = 'Analytics/object_detection/cfg/yolov3.cfg'

weightfile = 'Analytics/object_detection/weights/yolov3_weights.tf'

img_path = "Analytics/object_detection/data/images/person.jpg"

model = YOLOv3Net(cfgfile,model_size,num_classes)
model.load_weights(weightfile)

# class_names = load_class_names(class_name)

# gender and age detector

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

if os.path.exists('Analytics/gender_age/pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5'):
  weight_file = os.getcwd() + '/Analytics/gender_age/pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5'
else:
  print("the file doesn't exist")
  weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="Analytics/gender_age/pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
age_gen_model = get_model(cfg)
age_gen_model.load_weights(weight_file)

def runAnalytic(request,video_id):
    video = get_object_or_404(Video,id = video_id)
    context ={
        'video':video,
    }
    print(video.video.url)
    test_img()
    # test_video(video.video.url)
    return render(request,'analytic.html',context)

def index(request):
  img = cv2.imread('Analytics/data/example.jpg')
  image,preds = mainDriver.main(img,model,age_gen_model,img_size,True)
  print(image)
  im = Image.fromarray(image)
  buffered = BytesIO()
  im.save(buffered, format="jpeg")
  im_bytes = buffered.getvalue()  # im_bytes: image in binary format.
  im_b64 = base64.b64encode(im_bytes)
  im_b64 = "data:image/jpeg;charset=utf-8;base64" + (str(im_b64).replace("b'","")[:-1])
  print(im_b64)
  cv2.imwrite("Analytics/results/result.jpg",image)
  return render(request,'index.html')

def test_img():
  img = cv2.imread('Analytics/data/example.jpg')
  print(img)
  boxes,scores, classes, nums,class_names,preds = mainDriver.main(img,model,age_gen_model,img_size,draw_output=False)
  print("boxes:")
  print(boxes)
  print("scores:")
  print(scores)
  print("classes:")
  print(classes)
  print("nums:")
  print(nums)
  print("class_names:")
  print(class_names)
  print("preds:")
  print(preds)

  # cv2.imwrite("output.png",res)

def test_video(videopath):
  videopath = videopath[1:]
  print(videopath)
  cap= cv2.VideoCapture(videopath)
  cap.set(cv2.CAP_PROP_FPS, 10)
  print(cap.get(cv2.CAP_PROP_FPS))
  preds = []
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      boxes,scores, classes, nums,class_names,preds = mainDriver.main(frame,model,age_gen_model,img_size,draw_output=False)
      preds.append(preds)
  cap.release()
  cv2.destroyAllWindows()
  return HttpResponse(preds)
  # out = cv2.VideoWriter('output-trial-clip4.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 50, size)
  # for i in range(len(img_array)):
  #     out.write(img_array[i])
  # out.release()
