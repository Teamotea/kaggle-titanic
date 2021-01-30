import sys
import xgboost as xgb
from logs.logger import get_logger
import os

sys.path.append('')

class ModelXGB:
    def __init__(self, num_round=10, logging=False, verbose_eval=True):
        self.model = None
        self.num_round = num_round
        self.verbose_eval = verbose_eval
        exp_version = os.getenv('exp_version')
        model_name = os.getenv('model_name')
        if logging:
            logger = get_logger(exp_version)
            logger.info('=== XGB MODEL LOGGING STARTED ===')
            logger.info(f'EXP VERSION: {exp_version}')
            logger.info(f'MODEL NAME: {model_name}')
    
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