from django.http.response import HttpResponse
from django.shortcuts import render
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont
import base64
import cv2
from people_counter import mainDriver
import numpy as np
import tensorflow as tf
from people_counter.object_detection.utils import load_class_names, output_boxes, draw_outputs, resize_image
from people_counter.object_detection.yolov3 import YOLOv3Net
from pathlib import Path
import dlib
import os
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from people_counter.gender_age.src.factory import get_model
from people_counter.gender_age.get_gender_age import main_driver

# global settings.
model_size = (416, 416,3)
num_classes = 80

class_name = 'people_counter/object_detection/data/coco.names'
max_output_size = 40
max_output_size_per_class= 20

iou_threshold = 0.5

confidence_threshold = 0.5

cfgfile = 'people_counter/object_detection/cfg/yolov3.cfg'

weightfile = 'people_counter/object_detection/weights/yolov3_weights.tf'

img_path = "people_counter/object_detection/data/images/person.jpg"

model = YOLOv3Net(cfgfile,model_size,num_classes)
model.load_weights(weightfile)

# class_names = load_class_names(class_name)

# gender and age detector

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

if os.path.exists('people_counter/gender_age/pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5'):
  weight_file = os.getcwd() + '/people_counter/gender_age/pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5'
else:
  print("the file doesn't exist")
  weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="people_counter/gender_age/pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
model_name, img_size = Path(weight_file).stem.split("_")[:2]
img_size = int(img_size)
cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
age_gen_model = get_model(cfg)
age_gen_model.load_weights(weight_file)


def index(request):
  img = cv2.imread('people_counter/data/diverse-group-of-people.jpg')
  image,preds = mainDriver.main(img,model,age_gen_model,img_size,True)
  # print(image)
  im = Image.fromarray(image)
  buffered = BytesIO()
  im.save(buffered, format="jpeg")
  im_bytes = buffered.getvalue()  # im_bytes: image in binary format.
  im_b64 = base64.b64encode(im_bytes)
  im_b64 = "data:image/jpeg;charset=utf-8;base64" + (str(im_b64).replace("b'","")[:-1])
  print(im_b64)
  # cv2.imwrite("people_counter/results/result.jpg",image)
  return render(request,'index.html')

# def test_video():
#   videopath='test-videos/trial-clip3.mov'
#   cap= cv2.VideoCapture(videopath)
#   cap.set(cv2.CAP_PROP_FPS, 10)
#   print(cap.get(cv2.CAP_PROP_FPS))
#   img_array = []
#   size = ()
#   while(cap.isOpened()):
#       ret, frame = cap.read()
#       if ret == False:
#           break
#       img,preds = mainDriver.main(frame,model,age_gen_model,img_size,draw_output=True)
#       # cv2.imwrite('kang'+str(i)+'.jpg',frame)
#       height, width, layers = img.shape
#       size = (width,height)
#       img_array.append(img)
#       # cv2.waitKey(100)
#   cap.release()
#   cv2.destroyAllWindows()
#   out = cv2.VideoWriter('output-trial-clip4.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 50, size)
#   for i in range(len(img_array)):
#       out.write(img_array[i])
#   out.release()
