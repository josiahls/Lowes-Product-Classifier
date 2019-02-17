from time import time

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from util import *
from tensorflow.python.keras.applications import InceptionResNetV2, MobileNet
from tensorflow.python.keras.applications import NASNetMobile
from tensorflow.python.keras.applications import VGG19
from Logging import *
from sklearn.utils import class_weight
from Config import Config

DATASET_LOADED = False
STANDARD_IMAGE_SIZE = (224, 224, 3)
X = None
Y = []

"""
https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/bayesian_neural_network.py

"""

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def load_images(n_classes, data_dir, target_dict):
    global X, Y
    print(f'Loading Images')
    # For each class
    for directory in Config.DATA_DIR_NAMES:
        # For each image in that class
        for filename in os.listdir(data_dir + directory):
            # Get a standard sized image
            im = np.array(Image.open(data_dir + directory + os.sep + filename)
                          .resize(STANDARD_IMAGE_SIZE[0:2], Image.ANTIALIAS))
            # Get the actual class for that image
            Y.append(target_dict[directory])
            # Build the X sample array via stacking
            if X is not None:
                X = np.vstack((X, im[np.newaxis, ...]))
                # break
            else:
                X = np.array([im])

    # Fix X
    # Scale it, and flatten
    X = np.array([np.array(image / 255) for image in X])
    # Fix Y
    Y = np.array(Y).reshape(-1, 1)
    Y = to_categorical(Y, num_classes=n_classes)

    return X, Y


def train(hyper_params, reset_dataset=False):
    # Define upper level settings InceptionResNetV2
    global DATASET_LOADED, X, Y
    model_dir = f'./run_logs/{time()}_{"_".join("{!s}.{!r}".format(key,val) for (key,val) in hyper_params.items())}'

    # Get the directory path
    data_dir = get_absolute_data_path()
    target_dict = {name: i for i, name in enumerate(Config.DATA_DIR_NAMES)}

    if reset_dataset or not DATASET_LOADED:
        X, Y = load_images(hyper_params['n_classes'], data_dir, target_dict)
        DATASET_LOADED = True

    if hyper_params['sanity_test']:
        X_train, X_test, y_train, y_test = (X, X, Y, Y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

    print("Done building data set")
    model = None
    x = None
    if hyper_params['name'] == 'VGG19':
        model = VGG19(include_top=False, weights=None,
                      input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
        # If we want to use weights, then try to load them
        if hyper_params['use_weights']:
            model.load_weights('./weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

    elif hyper_params['name'] == 'InceptionResNetV2':
        model = InceptionResNetV2(include_top=False, weights=None,
                                  input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
        # If we want to use weights, then try to load them
        if hyper_params['use_weights']:
            model.load_weights('./weights/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')
    elif hyper_params['name'] == 'NASNetMobile':
        model = NASNetMobile(include_top=False, weights=None,
                             input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
        # If we want to use weights, then try to load them
        if hyper_params['use_weights']:
            model.load_weights('./weights/NASNet-mobile-no-top.h5')
    elif hyper_params['name'] == 'MobileNet':
        model = MobileNet(include_top=False, weights=None,
                          input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
        # If we want to use weights, then try to load them
        if hyper_params['use_weights']:
            model.load_weights('./weights/mobilenet_1_0_224_tf_no_top.h5')
    elif hyper_params['name'] == 'MobileNetBayesian':
        model = MobileNet(include_top=False, weights=None,
                          input_tensor=None, input_shape=STANDARD_IMAGE_SIZE, pooling=None)
        # If we want to use weights, then try to load them
        if hyper_params['use_weights']:
            model.load_weights('./weights/mobilenet_1_0_224_tf_no_top.h5')

    """ Freeze the previous layers """
    for layer in model.layers:
        layer.trainable = False
    """ By Setting top to False, we need to add our own classification layers """
    # The model documentation notes that this is the size of the classification block
    x = GlobalAveragePooling2D()(model.output)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(hyper_params['n_classes'], activation='softmax', name='predictions')(x)

    # create graph of your new model
    model = Model(inputs=model.inputs, outputs=x, name=hyper_params['name'])

    if hyper_params['opt'] == 'sgd':
        opt = SGD(lr=0.01)
    else:
        opt = 'adam'

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'mean_squared_error'])

    tensorboard = TrainValTensorBoard(log_dir=model_dir, histogram_freq=0, X_train=X_train,
                                      X_test=X_test, y_train=y_train, y_test=y_test,
                                      write_graph=True, write_images=False)

    """ Classes are going to be very imbalanced. Weight them """
    class_weights = class_weight.compute_class_weight('balanced', np.unique([y.argmax() for y in y_train]),
                                                      [y.argmax() for y in y_train])

    """ Add image augmentation """
    if hyper_params['use_aug']:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            channel_shift_range=0.5,
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.5, 1.0],
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1)
        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                            steps_per_epoch=len(X_train) / 32, epochs=hyper_params['epochs'],
                            validation_data=(X_test, y_test), callbacks=[tensorboard], class_weight=class_weights)
    else:
        model.fit(X_train, y_train, epochs=hyper_params['epochs'], validation_data=(X_test, y_test),
                  callbacks=[tensorboard], class_weight=class_weights)

    print(f'\nEvaluation: {model.evaluate(X_test, y_test)}')  # So this is currently: loss & accuracy
    prediction_y = model.predict(X_test)
    print(f'\nPrediction: {prediction_y}')
    print(f'\nFor Y targets {y_test}')

    # Save entire model to a HDF5 file
    model.save(model_dir + '/model.h5')

    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(prediction_y, axis=1))
    np.set_printoptions(precision=2)
    print(cnf_matrix)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=[c for c in target_dict],
    #                       title='Confusion matrix using ' + hyper_params['name'])

    # print(f'Saving confusion matrix to {model_dir + os.sep + "confusion_matrix.jpg"}')
    # plt.savefig(model_dir + os.sep + 'confusion_matrix.jpg')
    #
    # from sklearn.metrics import precision_recall_fscore_support
    # metrics = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(prediction_y, axis=1),
    #                                           average='weighted')
    # plot_precision_recall_f1(metrics, ['Precision', 'Recall', 'f Score'],
    #                          title='Metrics for ' + hyper_params['name'])
    # print(f'Saving confusion matrix to {model_dir + os.sep + "prec_recall_fscore.jpg"}')
    # plt.savefig(model_dir + os.sep + 'prec_recall_fscore.jpg')
    tf.keras.backend.clear_session()


def evaluate():
    pass


if __name__ == '__main__':
    epochs = 200
    classes = len(Config.DATA_DIR_NAMES)
    hyper_params = []
    # hyper_params.append({'epochs': epochs, 'name': 'VGG19', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'sgd', 'use_aug': True})
    # hyper_params.append({'epochs': epochs, 'name': 'InceptionResNetV2', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'sgd', 'use_aug': True})
    hyper_params.append({'epochs': epochs, 'name': 'NASNetMobile', 'n_classes': classes, 'use_weights': True,
                             'sanity_test': False, 'opt': 'adam', 'use_aug': True})
    # hyper_params.append({'epochs': epochs, 'name': 'VGG19', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'sgd', 'use_aug': False})
    # hyper_params.append({'epochs': epochs, 'name': 'InceptionResNetV2', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'sgd', 'use_aug': True})
    # hyper_params.append({'epochs': epochs, 'name': 'NASNetMobile', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'sgd', 'use_aug': False})
    # hyper_params.append({'epochs': epochs, 'name': 'VGG19', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'adam', 'use_aug': True})
    # hyper_params.append({'epochs': epochs, 'name': 'InceptionResNetV2', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'adam', 'use_aug': True})
    hyper_params.append({'epochs': epochs, 'name': 'NASNetMobile', 'n_classes': classes, 'use_weights': True,
                             'sanity_test': False, 'opt': 'adam', 'use_aug': True})
    # hyper_params.append({'epochs': epochs, 'name': 'VGG19', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'adam', 'use_aug': False})
    # hyper_params.append({'epochs': epochs, 'name': 'InceptionResNetV2', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'adam', 'use_aug': False})
    # hyper_params.append({'epochs': epochs, 'name': 'NASNetMobile', 'n_classes': classes, 'use_weights': True,
    #                          'sanity_test': False, 'opt': 'adam', 'use_aug': False})
    # hyper_params.append({'epochs': epochs, 'name': 'MobileNet', 'n_classes': classes, 'use_weights': True,
    #                      'sanity_test': False, 'opt': 'sgd', 'use_aug': True})
    hyper_params.append({'epochs': epochs, 'name': 'MobileNet', 'n_classes': classes, 'use_weights': True,
                         'sanity_test': False, 'opt': 'adam', 'use_aug': True})

    for params in hyper_params:
        # try:
        train(params)
        # except Exception:
        #     print(f'{params} failed')
