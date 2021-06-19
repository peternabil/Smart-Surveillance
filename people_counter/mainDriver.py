import tensorflow as tf
from people_counter.object_detection.utils import load_class_names, output_boxes, draw_outputs, resize_image,crop_persons
import cv2
import numpy as np
from pathlib import Path
import dlib
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from people_counter.gender_age.src.factory import get_model
from people_counter.gender_age.get_gender_age import detect_cropped_imgs

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
modhash = '6d7f7b7ced093a8b3ef6399163da6ece'

# weight_file = get_file("EfficientNetB3_224_weights.11-3.44.hdf5", pretrained_model, cache_subdir="pretrained_models",file_hash=modhash, cache_dir=str(Path(__file__).resolve().parent))


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

def main(img,model,gen_model,img_size,draw_output=False):
    # model = YOLOv3Net(cfgfile,model_size,num_classes)
    # model.load_weights(weightfile)
    class_names = load_class_names(class_name)

    # image = cv2.imread(img_path)
    image = img
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = resize_image(image, (model_size[0],model_size[1]))
    pred = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)
    persons = crop_persons(np.array(img), boxes, scores, classes, nums)
    print(persons)
    preds = detect_cropped_imgs(persons,img_size,gen_model)
    # print(preds)
    # if you want to draw the output on the image
    if draw_output:
        image = np.squeeze(image)
        img,person_num = draw_outputs(image, boxes, scores, classes, nums, class_names)
        cv2.putText(img, str(person_num)+" Persons", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (252, 244, 3), 2, cv2.LINE_AA)
        avg_age = 0
        f_num = 0
        m_num = 0
        for pred in preds:
            avg_age = avg_age + int(pred[0])
            if pred[1] == 'F':
                f_num = f_num + 1
            else:
                m_num = m_num + 1
        try:
            avg_age = avg_age / len(preds)
        except:
            avg_age = 0
        cv2.putText(img, str(f_num)+" Females, "+str(m_num)+" Males", (10,img.shape[0]-150), cv2.FONT_HERSHEY_SIMPLEX, 2, (252, 244, 3), 2, cv2.LINE_AA)
        cv2.putText(img, "Average age = " + str(int(round(avg_age, 0))), (10,img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (252, 244, 3), 2, cv2.LINE_AA)

        return image,preds
    # win_name = 'Image detection'
    else:
        return boxes,scores, classes, nums,class_names,preds

    # if you want to show the image
    # cv2.imshow(win_name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #If you want to save the result, uncommnent the line below:
    #cv2.imwrite('test.jpg', img)

# if __name__ == '__main__':
#     main()
