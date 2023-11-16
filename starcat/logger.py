import logging
import os
import time
from functools import wraps

enable_logging = True
# log_file_path = '/home/shenyueyue/Projects/starcat/log/runtime.log'  # ranku & dandelion
log_file_path = '/Users/sara/PycharmProjects/starcat/log/runtime.log'  # local

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
                log_file_name=log_file_path,
                level=logging.INFO
            )
            # time_logger.info(
            #     f"{os.path.basename(func.__code__.co_filename)} {func.__name__} {run_time:.6f} s"
            # )
            if len(args) == 0:
                file_name = os.path.basename(func.__code__.co_filename)  # 获取函数所在的文件名
                time_logger.info(
                    f"{file_name} {func.__name__} {run_time:.6f} s"
                )
            else:
                class_name = args[0].__class__.__name__  # 获取方法所属的类名
                time_logger.info(
                    f"{class_name} {func.__name__} {run_time:.6f} s"
                )

                if func.__name__ == '__call__':
                    test_log = args[4]
                    if test_log is True:
                        logage_log, mh_log, dm_log, Av_log, fb_log = args[1]
                        nstars_log = args[2]
                        acrate_log = result[1]
                        total_log = result[2]
                        sptime_log = result[3]
                        time_logger.info(
                            f'\nlogage={logage_log}, [M/H]={mh_log}, DM={dm_log}, Av={Av_log}, fb={fb_log}\n'
                            f'number of stars={nstars_log}, total sample={total_log}\n'
                            f'accepted rate={acrate_log}, sample times={sptime_log}\n\n'
                        )
            return result
        else:
            return func(*args, **kwargs)

    return wrapper
