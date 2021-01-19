import os
import logging
import pandas as pd
import numpy as np
from typing import Type, Tuple

from tqdm import tqdm

from utils.ops import get_state

import agent


def train_model(agent: agent, episode: int, data: list, window_size: int = 10, ep_count: int = 100, batch_size: int = 32):
    data = data.drop(columns="game_date")
    data_length = len(data["FD_points"].iloc[0]) - 1

    total_reward = []
    avg_loss = []

    state_data = _prep_data_for_state(data, 0, window_size + 1)
    state = get_state(state_data, 0, window_size + 1)

    for t in tqdm(range(data_length - 1), total=data_length, leave=True, desc=f"Episode {episode}/{ep_count}"):
        reward = 0
        starter_mask = [points[t + 1] > 0 for points in data["FD_points"]]
        next_state = get_state(state_data[starter_mask], t + 1, window_size + 1)
        done = t == data_length - 1

        # select a team
        action, prob = agent.act(state)

        reward = np.sum([data.iloc[player]["FD_points"][t] for player in action])
        total_reward.append(reward)
        agent.remember(state, action, reward, next_state, done)

        if done:
            agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, np.mean(total_reward), np.mean(np.array(avg_loss)))


def evaluate_model(agent: agent, data: list, debug: bool, date: list, window_size: int = 10) -> Tuple[float, list]:
    data = data.drop(columns="game_date")
    data_length = len(data["FD_points"].iloc[0]) - 1

    total_reward = []

    history = []
    
    state_data = _prep_data_for_state(data, 0, window_size + 1)
    state = get_state(state_data, 0, window_size + 1)

    for t in range(data_length):        
        reward = 0
        starter_mask = [points[t + 1] > 0 for points in data["FD_points"]]
        next_state = get_state(state_data[starter_mask], t + 1, window_size + 1)
        
        # select an action
        action, prob = agent.act(state, is_eval=True)

        reward_list = np.sum([data.iloc[player]["FD_points"][t] for player in action])
        reward = np.sum(reward_list)
        total_reward.append(reward)

        history.append((action, reward, prob, date[t]))

        if debug:
            logging.debug(f"Reward: {reward}")

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return np.mean(total_reward), history


def _prep_data_for_state(data: pd.DataFrame, t: int, window_size: int) -> pd.DataFrame:
    d = t - window_size if window_size > t else 0
    state_data = pd.DataFrame(index=data.index, columns=data.keys())
    for player, stats in data.iterrows():
        temp_stats = stats.reset_index(name="values")
        for _, values in temp_stats.iterrows():
            state_data.loc[player, values[0]] = np.mean(values[1][t : d])
    return state_data
