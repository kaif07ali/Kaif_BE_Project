import ntpath
import os
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './weights/discbrake-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_list('images', './data/Input/input.jpg', 'path to input image')
flags.DEFINE_string('output', './data/Output/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show image output')
flags.DEFINE_boolean('zip', False, 'pass zip file')


def main(_argv):
    df = pd.DataFrame(columns=['filename', 'predlabel', 'confidence', 'count'])
    input_size = FLAGS.size
    images = FLAGS.images
    if FLAGS.zip:
        images = os.listdir(images[0])
        images = ["./data/Input/" + x for x in images]
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        imgname = ntpath.basename(image_path)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        err_detected = int(pred_bbox[3][0])
        conf_score = max(pred_bbox[1][0])
        pred_label = int(pred_bbox[2][0][0])

        if (err_detected > 0):
            pred_label = 1

        if not FLAGS.zip:
            print("\nImage: ", imgname)
            print("Errors Detected: ", err_detected)
            if (pred_bbox[3][0] != 0):
                print("Confidence Score: ", conf_score)
            if (pred_label == 1):
                print("Verdict: Defective")
            elif (pred_label == 0):
                print("Verdict: OK!")

        df = df.append({'filename': imgname, 'predlabel': pred_label, 'confidence': conf_score, 'count': err_detected},
                       ignore_index=True)

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

        image = Image.fromarray(image.astype(np.uint8))
        if not FLAGS.dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(FLAGS.output + imgname, image)
    df.to_csv('./data/Output/ZipResults.csv')


if __name__ == '__main__':

    try:
        app.run(main)

    except SystemExit:
        pass
