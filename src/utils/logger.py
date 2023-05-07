#!/usr/bin/python
# -*- encoding: utf-8 -*-

import time
import logging
from pathlib import Path

import torch.distributed as dist


def setup_logger(logpth):
    logpth = Path(logpth)
    Path.mkdir(logpth, parents=True, exist_ok=True)
    logfile = "CABiNet-{}.log".format(time.strftime("%Y-%m-%d-%H-%M-%S"))
    logfile = logpth / logfile
    FORMAT = "%(message)s"
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank() == 0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=str(logfile))
    logging.root.addHandler(logging.StreamHandler())
