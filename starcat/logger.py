import logging
import os.path
import time
from functools import wraps

# formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
runtime_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def init_logger(logger_name, formatter, log_file_name, level):
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # determines whether the handler has been created
    if not logger.handlers:
        logger.addHandler(init_fileHd(formatter, log_file_name))
    return logger


def init_fileHd(myformatter, log_file_name):
    fileHd = logging.FileHandler(log_file_name)
    fileHd.setFormatter(myformatter)
    return fileHd


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        time_logger = init_logger(
            'time_logger',
            runtime_formatter,
            log_file_name='/home/shenyueyue/Projects/starcat/log/runtime.log',
            level=logging.WARNING
        )
        time_logger.info(
            f"{os.path.basename(func.__code__.co_filename)} {func.__name__} {run_time:.4f} s"
        )
        return result

    return wrapper
