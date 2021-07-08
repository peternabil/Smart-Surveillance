from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from Analytics.gender_age.src.factory import get_model
import PIL


def main_driver(img,model,detector,img_size):
    # image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()
    # print(model.summary())
    margin = 0
    predictions = []
    # for img in imgs:
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(input_img.shape)
    img_h, img_w, _ = input_img.shape
    
    # detect faces using dlib detector
    detected = detector(input_img, 1)
    # print(detected)
    faces = np.empty((len(detected), img_size, img_size, 3))
    
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

        # predict ages and genders of the detected faces
        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        # draw results
        for i, d in enumerate(detected):
            label = "{}, {}".format(int(predicted_ages[i]),"M" if predicted_genders[i][0] < 0.5 else "F")
            predictions.append([int(predicted_ages[i]),"M" if predicted_genders[i][0] < 0.5 else "F"])

    return predictions

def detect_cropped_imgs(persons,img_size,model):
    predictions = []
    for person in persons:
        person_img = cv2.imread('Analytics/object_detection/persons-imgs/'+person)
        print(img_size)
        person_img = cv2.resize(person_img,(img_size,img_size))
        use_arr = np.array([person_img])
        # print(use_arr.shape)
        
        pred = model.predict(use_arr)

        predicted_gender = pred[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_age = pred[1].dot(ages).flatten()
        # label = "{}, {}".format(int(predicted_age[0]),"M" if predicted_gender[0][0] < 0.5 else "F")
        predictions.append([int(predicted_age[0]),"M" if predicted_gender[0][0] < 0.5 else "F"])
    return predictions







# if __name__ == '__main__':
#     weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))

#     # for face detection
#     # detector = dlib.get_frontal_face_detector()

#     # load model and weights
#     model_name, img_size = Path(weight_file).stem.split("_")[:2]
#     img_size = int(img_size)
#     cfg = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={img_size}"])
#     model = get_model(cfg)
#     model.load_weights(weight_file)
#     imgs = [cv2.imread('imgs/img1.jpg'),cv2.imread('imgs/img2.jpg'),cv2.imread('imgs/img3.jpg')]
#     print(main_driver(imgs,model,detector))
