import numpy as np
import os
from logs.logger import get_logger

exp_version = os.getenv('exp_version')


def get_pred_result(model, train_x, train_y, test_x):
    from sklearn.model_selection import KFold
    preds = []
    preds_test = []
    va_idxes = []

    kf = KFold(n_splits=4, shuffle=True, random_state=1)

    for tr_idx, va_idx in kf.split(train_x):
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        model.fit(tr_x, tr_y, va_x, va_y)
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    return pred_train, preds_test


def get_acc_and_logloss(pred_y, train_y, logging=False):
    from sklearn.metrics import log_loss, accuracy_score
    pred_cat_train = [1 if pred > 0.5 else 0 for pred in pred_y]
    accuracy_score = accuracy_score(train_y, pred_cat_train)
    log_loss = log_loss(train_y, pred_y, eps=1e-7)

    if logging:
        logger = get_logger(exp_version)
        logger.info(f'ACCURACY: {accuracy_score}')
        logger.info(f'LOGLOSS: {log_loss}')
        logger.info(f'data size: {len(train_y)}')
        logger.info(f'correct predictions: {round(len(train_y) * accuracy_score)}')
        logger.info('')
    else:
        print(f'data size: {len(train_y)}')
        print(f'correct predictions: {int(len(train_y) * accuracy_score)}')
        print(f'accuracy: {accuracy_score:.10f}')
        print(f'logloss: {log_loss:.10f}')


def print_conf_matrix(pred_y, train_y):
    from sklearn.metrics import confusion_matrix
    pred_cat_train = [1 if pred > 0.5 else 0 for pred in pred_y]
    conf_matrix = confusion_matrix(train_y, pred_cat_train)
    print(conf_matrix)
