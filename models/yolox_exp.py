#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from src.yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.data_num_workers = 4
        # factor of model depth
        # self.depth = 1.00
        # factor of model width
        # self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        self.input_size = (736,1280)
        self.test_size = (736,1280)
        self.save_history_ckpt = False
        #self.multiscale_range = 5
        self.random_size = (14, 26)
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        # if set to 1, user could see log every iteration.
        self.print_interval = 5
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5
        self.weight_decay = 5e-4
        self.momentum = 0.9
        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0.8
        # prob of applying mixup aug
        self.mixup_prob = 0.0
        # prob of applying hsv aug
        self.hsv_prob = 0.6
        # prob of applying flip aug
        self.flip_prob = 0.5
        
        self.degrees = 0.0
        self.translate = 0.05
        self.scale = (0.01, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 0.01
        self.perspective = 0.0
        self.enable_mixup = False
