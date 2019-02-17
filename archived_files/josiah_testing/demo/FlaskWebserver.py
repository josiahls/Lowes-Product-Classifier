#!/usr/bin/env python
import skimage
import sklearn
import numpy as np
import os
from flask import Flask, render_template, Response, url_for
from demo.CameraCapture import MyVideoCapture
import cv2
from tensorflow.python.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image
from util import *
import pandas as pd
from collections import Counter

DATA_DIR_NAMES = ['FEIT_40W_T8_TUBE_MCRWV_BULB_120V',
                  'GE_60W_LED_A19_FROST_5000K_8CT',
                  'GE_Appliance_LED_40W_Warm_White',
                  'GE_Basic_LED_60W_Soft_Light',
                  'GE_Basic_LED_90W_Daylight',
                  'GE_Classic_LED_65W_Soft_White',
                  'GE_Vintage_LED_60W_Warm_Light',
                  'OSI_60W_13W_CFL_SOFT_WHITE_6_CT']

app = Flask(__name__)

name_to_load = ['FEIT_40W_T8_TUBE_MCRWV_BULB_120V'] * 5

# loaded_image_locations = pd.read_csv(os.path.join(get_absolute_data_path()[:-5], 'josiah_testing', 'demo',
#                                                   './Lowes Display Sheet - Sheet1.csv'))

tf.keras.backend.clear_session()
STANDARD_IMAGE_SIZE = (224, 224, 3)
model = InceptionResNetV2(include_top=False, weights=None,
                          input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
x = Flatten(name='flatten')(model.output)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(len(DATA_DIR_NAMES), activation='softmax', name='predictions')(x)

# create graph of your new model
model = Model(inputs=model.inputs, outputs=x, name='InceptionResNetV2')
# print(model.summary())

weights_path = os.path.join(get_absolute_data_path()[:-5], 'josiah_testing', 'demo', 'model.h5')
# If we want to use weights, then try to load them
model.load_weights(weights_path)
global graph
graph = tf.get_default_graph()


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        _, byte_frame, frame = camera.get_frame()
        resized_image = np.array(Image.fromarray(frame).resize(STANDARD_IMAGE_SIZE[0:2], Image.ANTIALIAS))
        with graph.as_default():
            id = np.argmax(model.predict(np.expand_dims(np.array(resized_image / 255), axis=0))[0])
            print(DATA_DIR_NAMES[id])
            global name_to_load
            name_to_load.pop(0)
            name_to_load.append(DATA_DIR_NAMES[id])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n')


def get_image():
    while True:
        b = Counter(name_to_load)
        name = b.most_common(1)[0][0]
        frame = skimage.io.imread(os.path.abspath("demo") + f'/static/thumbnail_images/{name}.png')
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(MyVideoCapture(video_source=0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/classification_image')
def classification_image():
    return Response(get_image(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
