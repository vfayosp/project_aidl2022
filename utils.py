import math
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class SummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        super().__init__(log_dir, comment, **kwargs)

    def add_scalar_dict(self, dictionary, global_step, tag=None):
        for name, val in dictionary.items():
            if tag is not None:
                name = os.path.join(tag, name)
            self.add_scalar(name, val, global_step)


def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0


def getNDCG(recommend_list, gt_item):
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2) / math.log(idx + 2)
    else:
        return 0
