# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class TensorBoardLogger:
    def __init__(self, log_dir, experiment_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag, tag_scalar_dict, step):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        self.writer.close()
