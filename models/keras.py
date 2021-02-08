from keras import Sequential
from logs.logger import get_logger
import os


class ModelKeras:
    def __init__(self, logging):
        self.model = Sequential()
        self.batch_size = None
        self.epochs = None
        self.verbose = None
        exp_version = os.getenv('exp_version')
        if logging:
            logger = get_logger(exp_version)
            logger.info('=== NN KERAS MODEL ===')

    def add_layers(self, *layers):
        for layer in layers:
            self.model.add(layer)

    def compile(self, optimizer='adam', metrics=['accuracy']):
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    def set_fit_params(self, batch_size=None, epoch=1, verbose=1):
        self.batch_size = batch_size
        self.epochs = epoch
        self.verbose = verbose

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(va_x, va_y), verbose=self.verbose)

    def predict(self, x):
        pred = self.model.predict(x)
        return pred
