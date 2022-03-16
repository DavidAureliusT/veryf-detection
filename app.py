from flask import Flask
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = '.\\Tesseract-OCR\\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = "/app/.apt/usr/bin/tesseract"
import json
import warnings
warnings.filterwarnings('ignore') 
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/get_invoice_number")
def get_invoice_number():
    print("===VERYF Detection===")
    start_time = time.time()
    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    IMAGES_DIR          = "./images/"
    IMAGE_FILENAME      = "sample2.jpg"
    LABEL_MAP_PATH      = "./data/label_map.pbtxt"
    SAVED_MODEL_PATH    = "./data/saved_model"

    print("--- %s seconds : Enable GPU memory allocation ---" % (start_time - time.time()))

    # 1. Load the Model
    detect_fn = tf.saved_model.load(SAVED_MODEL_PATH)

    # 2. Load label map data (for plotting)
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True) 

    # 3. Convert image into numpy_array
    def load_image_into_numpy_array(path):
        return np.array(Image.open(path))
    image_np = load_image_into_numpy_array(IMAGES_DIR + IMAGE_FILENAME) 

    # 4. The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)

    # 5. The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # 6. Run Detection
    detections = detect_fn(input_tensor)

    print("--- %s seconds : Detection by ResNet50_v2 ---" % (start_time - time.time()))

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

    print("--- %s seconds : Draw Bounding Box ---" % (start_time - time.time()))

    # 10. Cropping Invoice Number Box
    im_crop = image_pil.crop((left, top, right, bottom))
    crop_image_filename = IMAGE_FILENAME.split(".")[0] + "_cropped.jpeg"
    im_crop.save(IMAGES_DIR + crop_image_filename, "JPEG")

    print("--- %s seconds : Image Cropped ---" % (start_time - time.time()))

    # 11. Extract Invoice Number using Tesseract OCR
    def resize_image(filename):
        low_res_img_pil = Image.open(IMAGES_DIR + filename)
        h, w = low_res_img_pil.size
        resize = h*4, w*4
        im_resized = low_res_img_pil.resize(resize, Image.ANTIALIAS)
        resized_img_filename =  filename.split(".")[0] + "_resized.jpeg"
        im_resized.save(IMAGES_DIR + resized_img_filename, "JPEG")
        print("--- %s seconds : Image Resized ---" % (start_time - time.time()))
        return resized_img_filename

    def denoise_image(filename):
        noised_img_cv = cv2.imread(IMAGES_DIR + filename)
        denoised_img = cv2.fastNlMeansDenoising(noised_img_cv, None, 10)
        denoised_img_2x = cv2.fastNlMeansDenoising(denoised_img, None, 10)
        denoised_img_filename =  filename.split(".")[0] + "_denoised_2x.jpeg"
        cv2.imwrite(IMAGES_DIR + denoised_img_filename, denoised_img_2x)
        print("--- %s seconds : Image Denoised ---" % (start_time - time.time()))
        return denoised_img_filename
        
    def sharpen_image(filename):
        unsharpen_img_cv = cv2.imread(IMAGES_DIR + filename)
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        sharpened_img = cv2.filter2D(src=unsharpen_img_cv, ddepth=-1, kernel=kernel)
        sharpened_img_filename =  filename.split(".")[0] + "_sharpened.jpeg"
        cv2.imwrite(IMAGES_DIR + sharpened_img_filename, sharpened_img)
        print("--- %s seconds : Image Sharped ---" % (start_time - time.time()))
        return sharpened_img_filename

    def scale_contrast_image(filename):
        image = cv2.imread(IMAGES_DIR + filename)
        new_image = np.zeros(image.shape, image.dtype)
        alpha = 2.0 # Simple contrast control
        beta = 1    # Simple brightness control
        
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
        new_filename = filename.split(".")[0] + "_contrast.jpeg"
        cv2.imwrite(IMAGES_DIR + new_filename, new_image)
        print("--- %s seconds : Image's Contrast Gained ---" % (start_time - time.time()))
        return new_filename

    def adaptive_thresholding_image(filename):
        image = cv2.imread(IMAGES_DIR + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,3)
        new_filename = filename.split(".")[0] + "_at.jpeg"
        cv2.imwrite(IMAGES_DIR + new_filename, new_image)
        print("--- %s seconds : Highlight invoice number from the image ---" % (start_time - time.time()))
        return new_filename

    resized_image = resize_image(crop_image_filename)
    sharpened_image = sharpen_image(resized_image)
    denoised_image = denoise_image(sharpened_image)
    contrast_image = scale_contrast_image(denoised_image)
    at_image = adaptive_thresholding_image(contrast_image)

    img_cv = cv2.imread(IMAGES_DIR + at_image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    ocr_result = []
    for x in range(4):
        img_cv = cv2.rotate(img_cv, cv2.ROTATE_90_CLOCKWISE)
        _results = pytesseract.image_to_string(img_cv, config="outputbase digits")
        _results = _results.split('\n')
        for _result in _results:
            if len(_result) == 5:
                ocr_result.append(_result)

    response = {
        "data" : ocr_result
    }
    print("--- %s seconds : Runtime Finish ---" % (start_time - time.time()))
    print()
    print("Return -> "+json.dumps(response))
    return json.dumps(response)

