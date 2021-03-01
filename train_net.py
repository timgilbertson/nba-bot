import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.metrics import r2_score

from get_player_stats import build_NN_stats


def train_team_optimization():
    predicted_points = train_player_model()
    actual_team_picked, actual_best_team, picked_players, best_players = zip(*predicted_points.groupby("game_date").apply(_pick_best_team))
    print(f"Team picked average: {np.mean(actual_team_picked)}")
    print(f"Best team average: {np.mean(actual_best_team)}")
    import pdb; pdb.set_trace()


def train_player_model():
    X_train, X_test, y_train, y_test = build_NN_stats()
    model = _model(X_train.shape[1] - 4)
    loss = model.fit(np.array(X_train.drop(columns=["name", "positions", "game_date", "salary"])), np.array(y_train), epochs=5, verbose=1).history["loss"][0]
    prediction = model.predict(np.array(X_test.drop(columns=["name", "positions", "game_date", "salary"])))
    r2 = r2_score(y_test, prediction)
    print(r2)

    return X_test.assign(predicted_points=prediction, actual_points=y_test)


def _pick_best_team(game_date: pd.DataFrame) -> pd.DataFrame:
    
    predicted_points = game_date.sort_values("predicted_points", ascending=False).drop_duplicates(["name"])
    actual_points = game_date.sort_values("actual_points", ascending=False).drop_duplicates(["name"])
    picked_team, picked_players = _optimizer(predicted_points)
    best_team = best_players = _optimizer(actual_points)    

    return (np.sum(picked_team), np.sum(best_team), picked_players, best_players)


def _optimizer(sorted_by_points: pd.DataFrame):
    positions = []
    salary = []
    points = []
    players = []
    for i, row in sorted_by_points.iterrows():
        if len(positions) == 9:
            break

        if row["positions"] == "center":
            position_filled = True if np.sum([pos == row["positions"] for pos in positions]) == 1 else False
        else:
            position_filled = True if np.sum([pos == row["positions"] for pos in positions]) == 2 else False

        free_salary = np.sum(salary) + row["salary"] <= 60000
        if not position_filled and free_salary:
            positions.append(row["positions"])
            salary.append(row["salary"])
            points.append(row["actual_points"])
            players.append(row["name"])
    return points, players


def _model(shape):
    """
    Creates the neural net model
    """
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=shape))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=512, activation="tanh"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=1))

    model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))
    return model


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """
    Huber loss function
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


if __name__ == "__main__":
    train_team_optimization()