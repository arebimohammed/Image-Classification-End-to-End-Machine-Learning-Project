import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image_size(img, target_size=(150,150)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if img.size != (150,150):
        img = img.resize(target_size, Image.Resampling.NEAREST)
    return img


class Preprocessor:
    def __init__(self, target_size = (150,150)):
        self.target_size = target_size

    def resize_image(self, img):
        return prepare_image_size(img, self.target_size)

    def image_to_array(self, img):
        return np.array(img, dtype='float32')

    def convert_to_tensor(self, img):
        small = self.resize_image(img)
        x = self.image_to_array(small)
        X = np.expand_dims(x, axis=0)
        return X

    def from_path(self, path):
        with Image.open(path) as img:
            return self.convert_to_tensor(img)

    def from_url(self, url):
        img = download_image(url)
        return self.convert_to_tensor(img)



interpreter = tflite.Interpreter(model_path='intel-classifier.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

preprocessor = Preprocessor(target_size = (150,150))
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def predict(url):

    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    return dict(zip(classes, preds[0].tolist()))
    

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

