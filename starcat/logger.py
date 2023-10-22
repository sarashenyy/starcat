import logging
import time
from functools import wraps

# formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
runtime_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

enable_logging = False


def init_logger(logger_name, formatter, log_file_name, level):
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # determines whether the handler has been created
    if not logger.handlers:
        logger.addHandler(init_fileHd(formatter, log_file_name))
    return logger


def init_fileHd(myformatter, log_file_name):
    # fileHd = logging.FileHandler(log_file_name)
    fileHd = logging.FileHandler(log_file_name, mode='a')
    fileHd.setFormatter(myformatter)
    return fileHd


def log_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if enable_logging:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            run_time = end_time - start_time
            time_logger = init_logger(
                'time_logger',
                runtime_formatter,
                log_file_name='/home/shenyueyue/Projects/starcat/log/runtime.log',
                level=logging.INFO
            )
            # time_logger.info(
            #     f"{os.path.basename(func.__code__.co_filename)} {func.__name__} {run_time:.6f} s"
            # )
            class_name = args[0].__class__.__name__  # 获取方法所属的类名
            time_logger.info(
                f"{class_name} {func.__name__} {run_time:.6f} s"
            )
            return result
        else:
            return func(*args, **kwargs)

    return wrapper
