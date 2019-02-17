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
"""Handles date set importing as usable tensors. Defaults to train, test, validation splitting."""
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import helpers
import os
from functools import partial
from sklearn.model_selection import train_test_split
import tensorflow as tf
from absl import flags

# Can parse cmd arguments
flags.DEFINE_list("categories", default=['FEIT_40W_T8_TUBE_MCRWV_BULB_120V', 'GE_40W_RelaxLED',
                                         'GE_60W_LED_A19_FROST_5000K_8CT', 'GE_Appliance_LED_11W_Soft_White',
                                         'GE_Appliance_LED_40W_Warm_White', 'GE_Basic_LED_60W_Soft_Light',
                                         'GE_Basic_LED_90W_Daylight', 'GE_Classic_LED_65W_Soft_White',
                                         'GE_Vintage_LED_60W_Warm_Light', 'OSI_60W_13W_CFL_SOFT_WHITE_6_CT',
                                         'Theres_No_Bulb'], help="List of images in their particular order")

FLAGS = flags.FLAGS


def _parse_function(element, image_size):
    """
    Taken from https://www.tensorflow.org/guide/datasets

    :param filename:filename, label, label_string
    :param label:
    :param image_size:
    :return:
    """
    image_string = tf.read_file(element['images'])
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, image_size[:2])
    return image_resized, element['labels'], element['names']


def get_fake_data(image_shape: Tuple[float, float, float], num_classifications: int = 10) -> Tuple:
    """
    Used for quick sanity checks. Returns the train, test, and validation sets.

    :param image_shape: A 3 dimensional tuple. Expected in RGB format
    :param num_classifications: Don't make this too large. The NN will attempt to classify these.
                                if some are too random then this will not test well.
    :return:
    """
    x_train = np.random.rand(num_classifications, *image_shape).astype(np.float32)
    y_train = np.random.permutation(np.arange(num_classifications)).astype(np.int32)
    x_test = np.random.rand(num_classifications, *image_shape).astype(np.float32)
    y_test = np.random.permutation(np.arange(num_classifications)).astype(np.int32)
    x_validate = np.copy(x_test)
    y_validate = np.copy(y_test)

    return (x_train, y_train), (x_test, y_test), (x_validate, y_validate)


def get_real_data(image_shape: Tuple[float, float, float], data_dir: str = 'data',
                  set_distribution=None):
    if set_distribution is None:
        set_distribution = {'train': .7, 'test': .1, 'validation': 0.2}

    FLAGS(sys.argv)
    # Get the absolute path to the data folder
    absolute_path = helpers.get_absolute_data_path(data_dir)
    # Set up all of the images and the corresponding target
    X = None  # type: np.array()
    Y = None  # type: np.array()

    # Get list of already validated images if they exist
    try:
        test_df = pd.read_csv(os.path.join(absolute_path, 'test.csv'))
    except FileNotFoundError:
        test_df = None

    print(absolute_path)
    # Load images and their labels
    for dir in [_ for _ in tf.gfile.ListDirectory(absolute_path) if _ in FLAGS.categories]:
        image_dir = os.path.join(absolute_path, dir)
        print(dir)
        for image in tf.gfile.ListDirectory(image_dir):
            if X is None:
                print(image)
                X = np.array([str(image).lower()])
                Y = np.array([dir])
            else:
                X = np.hstack((X, np.array([str(image).lower()])))
                Y = np.hstack((Y, np.array([dir])))

    # Load the persistent validation data
    if test_df is None:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=int(len(X) * set_distribution['test']))
        test_list = {'filenames': x_test, 'class': y_test}
        test_df = pd.DataFrame(test_list)
        pd.DataFrame(test_df).to_csv(os.path.join(absolute_path, 'test.csv'))

    # Reset the X and Y to their smaller sets now
    x_train = [x for x in X if x not in test_df['filenames']]
    Y = [Y[X == x][0] for x in X if x not in test_df['filenames']]
    X = x_train
    x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size=int(len(X) *
                                                                                        set_distribution['validation']))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rotation_range=0,
        channel_shift_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=[0, .2],
        horizontal_flip=False,
        vertical_flip=False,
        validation_split=set_distribution['validation'],
        zoom_range=0.1)

    # test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #     # featurewise_center=True,
    #     # featurewise_std_normalization=True,
    #     # rotation_range=0,
    #     # channel_shift_range=0.0,
    #     # width_shift_range=0.0,
    #     # height_shift_range=0.0,
    #     # brightness_range=0,
    #     # horizontal_flip=False,
    #     # vertical_flip=False,
    #     validation_split=set_distribution['validation'],
    #     zoom_range=0)
    train_iter = datagen.flow_from_directory(absolute_path, subset='training')
    validation_iter = datagen.flow_from_directory(absolute_path, subset='validation')

    train_iter.filenames = [f for f in train_iter.filenames if not str(f).__contains__('._')]
    validation_iter.filenames = [f for f in validation_iter.filenames if not str(f).__contains__('._')]

    # Save the test dataset for persistent comparisons between runs
    return train_iter, validation_iter


if __name__ == '__main__':
    tf.enable_eager_execution()
    sess = tf.InteractiveSession()
    # Test fake data generator:
    print(get_real_data((224, 224, 3)))
    sess.close()
