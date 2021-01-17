import os
import math
import logging
import pandas as pd

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1
    res_list = []
    for column in data:
        column_list = data[column].to_list()
        block = column_list[d: t + 1] if d >= 0 else -d * [column_list[0]] + column_list[0: t + 1]  # pad with t0
        res = []
        for i in range(n_days - 1):
            res.append(sigmoid(block[i + 1] - block[i]))
        res_list.append(res)
    return np.array(res_list)


def _calc_moving_average(data: pd.DataFrame):
    import pdb; pdb.set_trace()