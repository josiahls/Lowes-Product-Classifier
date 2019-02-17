from collections import Counter
from logging import debug

import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.applications import MobileNet

from Config import Config
from util import *


class MobileNetModelWrapper:
    def __init__(self, recent_log=None):
        # Init constants
        # Init classification buffer. Designed to smooth the classification
        self.name_to_load = [Config.DATA_DIR_NAMES[0]] * 5
        self.name = 'MobileNet'

        tf.keras.backend.clear_session()
        self.STANDARD_IMAGE_SIZE = (224, 224, 3)
        model = MobileNet(include_top=False, weights=None,
                          input_tensor=None, input_shape=self.STANDARD_IMAGE_SIZE, pooling=None)
        x = GlobalAveragePooling2D()(model.output)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        x = Dense(len(Config.DATA_DIR_NAMES), activation='softmax', name='predictions')(x)

        # create graph of your new model
        self.model = Model(inputs=model.inputs, outputs=x, name='MobileNet')
        # print(model.summary())

        weights_path = os.path.join(get_absolute_data_path()[:-5], 'josiah_testing', 'run_logs')
        if recent_log is None:
            recent_log_dir = [_ for _ in os.listdir(weights_path) if str(_).lower().__contains__(self.name.lower())]
            recent_log_dir.sort(reverse=True)
        else:
            recent_log_dir = [recent_log]

        weights_path = os.path.join(weights_path, recent_log_dir[0], 'model.h5')
        print(f'Loading final weight path: {weights_path}')

        # If we want to use weights, then try to load them
        self.model.load_weights(weights_path)
        global graph
        graph = tf.get_default_graph()

    def predict(self, image: np.array) -> (str, int, float):
        """
        The predict method returns the string name, the id, and the confidence of that prediction.

        :param image: Must be a N X M X 3 numpy array representing an image
        :return: string name, the id, and the confidence of that prediction
        """
        resized_image = np.array(Image.fromarray(image).resize(self.STANDARD_IMAGE_SIZE[0:2], Image.ANTIALIAS))
        with graph.as_default():
            # Get the prediction scores
            scores = self.model.predict(np.expand_dims(np.array(resized_image / 255), axis=0))[0]
            # Select the best one
            class_id = np.argmax(scores).astype(np.int8)  # type: int
            debug(Config.DATA_DIR_NAMES[class_id])
            # Add the name to the queue of previous names.
            self.name_to_load.pop(0)
            self.name_to_load.append(Config.DATA_DIR_NAMES[class_id])

        # Smooth the name output.
        b = Counter(self.name_to_load)
        name = b.most_common(1)[0][0]
        return name, class_id, scores[class_id]
