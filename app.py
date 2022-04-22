from flask import Flask, jsonify, request
import time
import tensorflow as tf
# from object_detection.utils import label_map_util
import numpy as np
from PIL import Image, ImageDraw
import requests
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import warnings
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test", methods=["POST"])
def test():
    return jsonify({"data": request.values["data"] }) 

@app.route('/first')
def first():
    second_url = "http://127.0.0.1:5001/second"
    data = {"data": "data"}
    response = requests.post(second_url, data)
    return response.text

@app.route('/second', methods=["POST"])
def second():
    get_data = request.values["data"]
    # return {"count": get_data.count}
    return {"count": len(get_data)}


@app.route("/detect_invoice_number", methods=["POST"])
def get_invoice_number():
    start_time = time.time()
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # LABEL_MAP_PATH      = "./data/label_map.pbtxt"
    SAVED_MODEL_PATH    = "./data/saved_model"

    # 1. Load the Model
    detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)
    
    # 2. Load label map data (for plotting)
    # category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True) 
    
    # 3. Convert image into numpy_array
    image_np = np.array(Image.open(request.files['file'].stream))

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

    # 9. draw bounding box
    box = tuple(detections['detection_boxes'][0].tolist())
    ymin, xmin, ymax, xmax = box
    image_pil = Image.fromarray(np.uint8(image_np_with_detections)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height) # using normalize_coordinate
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=4, fill='red') # draw line

    # 10. Cropping Invoice Number Box
    im_crop = image_pil.crop((left, top, right, bottom))
    open_cv_image = np.array(im_crop)

    total_runtime = time.time() - start_time
    print("detection runtime: " + str(total_runtime)[:5] + " seconds")

    # ocr_url = "http://127.0.0.1:5002/get_invoice_number"
    test_url = "http://127.0.0.1:5002/cv_test"
    data = {
        "image": open_cv_image.tolist()
    }
    response = requests.post(test_url, data)

    return {
        "data": response.text,
    }
    # return {"data":open_cv_image.tolist()}
    # deploy to heroku


