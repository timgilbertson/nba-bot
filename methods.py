import os
import logging
import pandas as pd
import numpy as np
from typing import Type, Tuple
import random
from tqdm import tqdm

from utils.ops import get_state

import agent


def train_model(agent: agent, episode: int, data: list, window_size: int = 10, ep_count: int = 100, batch_size: int = 32):
    data_length = data.shape[2] - 1

    total_reward = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    for t in tqdm(random.sample(range(data_length - 1), 500), total=500, leave=True, desc=f"Episode {episode}/{ep_count}"):
        reward = 0
        next_state = get_state(data, t, window_size + 1)
        done = False

        # select a team
        actions, _ = agent.act(state)
        reward = np.sum(data[:, :, t + 1][actions, 11])
        if reward < 1:
            continue
        total_reward.append(reward)
        for action in actions:
            agent.remember(state, action, data[:, :, t + 1][action, 11], next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, np.mean(total_reward), np.mean(np.array(avg_loss)))


def evaluate_model(agent: agent, data: list, debug: bool, date: list, window_size: int = 10) -> Tuple[float, list]:
    data_length = data.shape[2] - 1

    total_reward = []

    history = []
    state = get_state(data, 0, window_size + 1)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t, window_size + 1)
        
        # select an action
        actions, prob = agent.act(state, is_eval=True)

        reward = np.sum(data[:, :, t][actions, 11])
        if reward < 1:
            continue
        total_reward.append(reward)

        history.append((actions, reward, prob, date[t]))

        if debug:
            logging.debug(f"Reward: {reward}")

        done = (t == data_length - 1)
        for action in actions:
            agent.memory.append((state, action, data[:, :, t + 1][action, 11], next_state, done))

        state = next_state

    return np.mean(total_reward), history
