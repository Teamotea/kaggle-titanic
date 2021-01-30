from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


def create_logger(exp_version):
    log_file = ('_{}.log'.fomat(exp_version)).resolve()

    # logger
    logger_ = getLogger(exp_version, mode="w")
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)