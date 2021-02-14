from keras import Sequential
from keras.optimizers import Adam
from logs.logger import get_logger
import os


class ModelKeras:
    def __init__(self, logging):
        self.model = Sequential()
        self.batch_size = None
        self.epochs = None
        self.verbose = None
        self.learning_rate = None
        self.status = {'COMPILE': False, 'SET_FIT_PARAMS': False}
        self.initial_weights = None
        exp_version = os.getenv('exp_version')
        if logging:
            logger = get_logger(exp_version)
            logger.info('=== NN KERAS MODEL ===')

    def add_layers(self, *layers):
        for layer in layers:
            self.model.add(layer)
        self.initial_weights = self.model.get_weights()

    def compile(self, learning_rate=0.001, metrics=['accuracy']):
        opt = Adam(learning_rate=learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)
        self.status.update({'COMPILE': True})

    def set_fit_params(self, batch_size=None, epochs=1, verbose=1):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.status.update({'SET_FIT_PARAMS': True})

    def fit(self, tr_x, tr_y, va_x, va_y, learning_rate=None, batch_size=None, epochs=None, verbose=None):
        if not self.status['COMPILE']:
            self.compile(learning_rate=learning_rate)
        if not self.status['SET_FIT_PARAMS']:
            self.set_fit_params(batch_size=batch_size, epochs=epochs, verbose=verbose)
        self.model.set_weights(self.initial_weights)
        # print(self.model.get_weights())
        self.model.fit(tr_x, tr_y, batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=(va_x, va_y), verbose=self.verbose)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred
