"""
Script for training Stock Trading Bot.

Usage:
  train.py [--strategy=<strategy>]
    [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] [--full-train]

Options:
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
  --full-train                      Whether to train the model on all data.
"""

import logging
import coloredlogs
from docopt import docopt
import pandas as pd
import numpy as np
import os
import keras.backend as K

from agent import Agent
from get_player_stats import get_player_stats
from methods import train_model, evaluate_model


def main(batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, full_train=False):
    """ Trains the nba bot using Deep Q-Learning.

    Args: [python train.py --help]
    """
    train_data, test_data, _, _ = get_player_stats()
    no_of_players = train_data.shape[0]

    agent = Agent(action_size=no_of_players, strategy=strategy, pretrained=pretrained, model_name=model_name)

    if full_train:
      train_data = train_data + test_data + test_data

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count, batch_size=batch_size)
        val_result, _ = evaluate_model(agent, test_data, debug, test_data[:, 10, :].tolist())
        _show_train_result(train_result, val_result)


def _switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logging.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _show_train_result(result, val_position, initial_offset = 0):
    """ 
    Displays training results
    """
    if val_position == initial_offset or val_position == 0.0:
        logging.info(f'Episode {result[0]}/{result[1]} - Train Reward: {result[2]:.2f}  Val Reward: USELESS  Train Loss: {result[3]:.4f}')
    else:
        logging.info(f'Episode {result[0]}/{result[1]} - Train Reward: {result[2]:.2f}  Val Reward: {val_position:.2f}  Train Loss: {result[3]:.4f})')


if __name__ == "__main__":
    args = docopt(__doc__)

    strategy = args["--strategy"]
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    full_train = args["--full-train"]

    coloredlogs.install(level="DEBUG")
    _switch_k_backend_device()

    try:
        main(batch_size=batch_size, ep_count=ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug, full_train=full_train)
    except KeyboardInterrupt:
        print("Aborted!")
