import xgboost as xgb

class ModelXGB:
    def __init__(self):
        self.model = None
    
    def fit(self, tr_x, tr_y, va_x, va_y, verbose=False):
        params = {'objective': 'binary:logistic', 'random_state': 1, 'eval_metric': 'logloss'}
        num_round = 10
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist, verbose_eval=False)
        print()
    
    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred