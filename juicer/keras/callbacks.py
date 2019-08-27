import json
import time

import keras


class JuicerCallback(keras.callbacks.Callback):
    def __init__(self, emit, job_id, task_id, epochs):
        super(JuicerCallback, self).__init__()
        self.emit = emit
        self.epoch_time_start = 0
        self.job_id = job_id
        self.epochs = epochs
        self.task_id = task_id

    def on_train_begin(self, logs=None):
        self.emit(name='update job',
                  identifier='',
                  id=self.job_id,
                  message='Training started',
                  status='RUNNING')

    def on_train_end(self, logs=None):
        self.emit(name='update job',
                  identifier='',
                  id=self.job_id,
                  message='Training ended',
                  status='RUNNING')

        self.emit(name='update job',
                  identifier='',
                  id=self.job_id,
                  message='Generating the classification report',
                  status='RUNNING')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
        self.emit(name='update job',
                  identifier='',
                  id=self.job_id,
                  message='Started epoch {} (zero-based, {}/{})'.format(
                      epoch, epoch + 1, self.epochs),
                  status='RUNNING',
                  )

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_time_start
        params = {"x": epoch, 'duration': duration}
        params.update(logs)
        msg = json.dumps(params)

        self.emit(name='update job',
                  id=self.job_id,
                  identifier='',
                  message='Epoch {} ended'.format(epoch),
                  status='RUNNING')
        self.emit(
            status='RESULTS',
            type='PROGRESS',
            name='update task',
            message=msg,
            identifier=self.task_id,
        )
