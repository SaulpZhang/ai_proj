import logging
from logging.handlers import RotatingFileHandler


def get_console_logger(name: str = "console_logger") -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"))
    logger_console = logging.getLogger(name)
    logger_console.setLevel(logging.INFO)
    logger_console.addHandler(console_handler)
    logger_console.propagate = False
    return logger_console

def get_file_logger(name: str = "file_logger") -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"))
    file_handler = RotatingFileHandler(
        "out.log", encoding="utf-8", maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"))


    logger_file = logging.getLogger(name)
    logger_file.setLevel(logging.INFO)
    logger_file.addHandler(console_handler)
    logger_file.addHandler(file_handler)
    logger_file.propagate = False

    return logger_file