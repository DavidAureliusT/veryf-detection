import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Init:
#   Variables:
#       - IMAGE_PATH : String -> .jpg || .jpeg file
#       - LABEL_MAP_PATH : String -> .pbtxt file
#       - SAVED_MODEL_PATH : String -> "reference to saved_model folder of exported model"

IMAGE_PATH          = "./sample.jpg"
LABEL_MAP_PATH      = "./label_map.pbtxt"
SAVED_MODEL_PATH    = "./saved_model"

# 1. Load the Model
detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)

# 2. Load label map data (for plotting)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True) 

# 3. Convert image into numpy_array
def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

image_np = load_image_into_numpy_array(IMAGE_PATH) 

# 4. The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)

# 5. The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# 6. Run Detection
detections = detect_fn(input_tensor)

# 7. All outputs are batches tensors.
#    Convert to numpy arrays, and take index [0] to remove the batch dimension.
#    We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# 8. detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
image_np_with_detections = image_np.copy()

# 9. manipulate output data
#    - image_np_with_detections         <-shape-> (img_h, img_w, 3)
#    - detections['detection_boxes']    <-shape-> (n, 4)
#    - detections['detection_scores']   <-shape-> (n)