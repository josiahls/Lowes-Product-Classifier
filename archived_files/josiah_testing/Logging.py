import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./run_logs', X_train=None, X_test=None,
                 y_train=None, y_test=None, **kwargs):
        """
        Found from:
        https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure

        This is useful for logging the test vs log data

        :param log_dir:
        :param kwargs:
        """
        # Make the original `TensorBoard` log to a subdirectory 'training'
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = X_test
        self.X_train = X_train
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

        """ Add the image progress """
        self.get_images()

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        # print(f'Logging for epoch {epoch} is {logs}')
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        """ Add the logs """
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        self._write_custom_summaries(epoch, logs)

    def _write_custom_summaries(self, step, logs=None):
        """Writes metrics out as custom scalar summaries.

        Arguments:
            step: the global step to use for Tensorboard.
            logs: dict. Keys are scalar summary names, values are
                NumPy scalars.

        """
        logs = logs or {}
        if context.executing_eagerly():
            # use summary ops
            with self.writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            # use FileWriter from v1 summary
            for name, value in logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, step)
        self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

    def get_images(self):
        print('Hello')
        image_shaped_input = tf.reshape(self.X_test, [-1, *self.X_test[0].shape])
        # image_shaped_input = tf.transpose(image_shaped_input, perm=[0, 3, 2, 1])
        return tf.summary.image(name='input', tensor=image_shaped_input, max_outputs=4)#len(self.X_test))