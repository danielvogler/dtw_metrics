"""Util functions."""
import datetime
import logging
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

if "/.venv/" in PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT[: PROJECT_ROOT.index("/.venv/")]


def initialize_logger(self, file_name: str):
    """Initialize logger.

    Args:
        file_name (_type_): file name to write log to
    """
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    file_name = os.path.splitext(file_name)[0]
    self.logger = logging.getLogger(file_name)
    logging.basicConfig(
        filename=self.logger_dir
        / str("log_-_" + file_name + "_-_" + current_time_str + ".log"),
        level=logging.INFO,
    )
