import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self,
        log_dir: str,
        name: str = 'breast_cancer',
        level: int = logging.INFO,
        tensorboard: bool = True
    ):
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'training.log'),
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        self.logger.addHandler(console_handler)

        self.tb_writer = SummaryWriter(log_dir=log_dir) if tensorboard else None

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step or 0)

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
