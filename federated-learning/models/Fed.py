#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


# FadeFedAvg
# global_w: current global weight
# new_local_w: newly arrived local weight
# fade_c: the fade coefficient for new_local_w
def FadeFedAvg(global_w, new_local_w, fade_c):
    w_avg = copy.deepcopy(global_w)
    for k in w_avg.keys():
        w_avg[k] += fade_c * new_local_w[k]
        w_avg[k] = torch.div(w_avg[k], 1 + fade_c)
    return w_avg
