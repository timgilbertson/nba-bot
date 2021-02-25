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
            return 1 - (1 / (1 + math.exp(x)))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        import pdb; pdb.set_trace()
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    d = t - n_days + 1 if t > n_days + 1 else 0
    res_list = []
    sliced_data = data[:, :, d : t + 1]
    for player in sliced_data.T:
        player = player.flatten().tolist()
        block = player + [player[0]]
        res = []
        for i in range(n_days - 1):
            # sigmoid_val = sigmoid(block[i + 1] - block[i]) if player[-1] > 0 else 0
            sigmoid_val = block[i + 1] - block[i]
            res.append(sigmoid_val)
        res_list.append(res)
    return np.array(res_list)
