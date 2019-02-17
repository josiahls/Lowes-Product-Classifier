# Copyright 2019 The Lowes UNCC group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================
"""MobileNet is an extremely small yet accurate model usable for mobile devices. """
import tensorflow as tf
from tensorflow.python.keras import Model
import datasets_handler
from tensorflow.python.keras.applications import InceptionResNetV2, MobileNet
from tensorflow.python.keras.applications import NASNetMobile
from tensorflow.python.keras.applications import VGG19
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MobileNetModel:
    def __init__(self, data_X, data_y):
        self.n_class = int(data_y.shape[0])
        self.model = None
        self._create_architecture(data_X, data_y)

    def _create_architecture(self, data_X, data_y):
        self.model = MobileNet(include_top=False, weights=None,
                               input_tensor=None, input_shape=list([int(_) for _ in data_X.shape[-3:]]), pooling=None)
        self.model.load_weights('./weights/mobilenet_1_0_224_tf_no_top.h5')

        """ Freeze the previous layers """
        for layer in self.model.layers:
            layer.trainable = False
        """ By Setting top to False, we need to add our own classification layers """
        # The model documentation notes that this is the size of the classification block
        x = GlobalAveragePooling2D()(self.model.output)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = Dropout(x, rate=0.5)
        # and a logistic layer -- let's say we have 200 classes
        x = Dense(int(data_y.shape[1]), activation='softmax', name='predictions')(x)
        # create graph of your new model
        self.model = Model(inputs=self.model.inputs, outputs=x, name='MobileNet')

        self.model.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy',
                           metrics=['accuracy', 'mean_squared_error'])

    def train(self, train_generator, validation_generator):
        print('Training Model')
        # fits the model on batches with real-time data augmentation:
        self.model.fit_generator(train_generator, steps_per_epoch=1, epochs=20, validation_steps=1,
                                 validation_data=validation_generator, verbose=1)


if __name__ == '__main__':
    tf.enable_eager_execution()
    train_generator, validation_generator = datasets_handler.get_real_data((224, 224, 3))
    x, y = train_generator.next()

    # Setup the model
    model = MobileNetModel(x, y)
    model.train(train_generator, validation_generator)
