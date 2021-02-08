import xgboost as xgb
import sys
from logs.logger import get_logger
import os


class ModelXGB:
    def __init__(self, model_name, logging, verbose_eval, num_round=10):
        self.model = None
        self.num_round = num_round
        self.verbose_eval = verbose_eval
        self.params_to_log = {'model_name': model_name, 'num_round': self.num_round}
        exp_version = os.getenv('exp_version')
        if logging:
            logger = get_logger(exp_version)
            logger.info('=== XGB MODEL ===')
            logger.info(f'PARAMS: {self.params_to_log}')

    def fit(self, tr_x, tr_y, va_x, va_y):
        params = {'objective': 'binary:logistic', 'random_state': 1, 'eval_metric': 'logloss'}
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, self.num_round, evals=watchlist, verbose_eval=self.verbose_eval)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred

    def get_model(self):
        return self.model


class ModelXGBSklearn:
    def __init__(self, model_name, logging, verbose, max_depth=None, n_estimators=10, learning_rate=0.3):
        # params at model creation
        self.model = xgb.XGBClassifier(n_estimators=n_estimators, random_state=1,
                                       learning_rate=learning_rate, max_depth=max_depth)
        # params at training
        self.verbose = verbose
        # params to log
        self.params_to_log = {'model_name': model_name, 'n_estimator': n_estimators,
                              'learning_rate': learning_rate, 'max_depth': max_depth}
        exp_version = os.getenv('exp_version')
        if logging:
            logger = get_logger(exp_version)
            logger.info('=== XGB SKLEARN MODEL ===')
            logger.info(f'PARAMS: {self.params_to_log}')

    def fit(self, tr_x, tr_y, va_x, va_y):
        self.model.fit(tr_x, tr_y, eval_metric='logloss', eval_set=[(tr_x, tr_y), (va_x, va_y)], verbose=self.verbose)

    def predict(self, x):
        return self.model.predict(x)

    def get_model(self):
        return self.model
