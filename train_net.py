import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam

from sklearn.metrics import r2_score

from get_player_stats import build_NN_player_stats


def train_team_optimization():
    predicted_points = train_player_model()
    actual_team_picked, actual_best_team = zip(*predicted_points.groupby("game_date").apply(_pick_best_team))
    import pdb; pdb.set_trace()


def train_player_model():
    X_train, X_test, y_train, y_test = build_NN_player_stats()
    model = _model(X_train.shape[1] - 2)
    loss = model.fit(np.array(X_train.drop(columns=["name", "game_date"])), np.array(y_train), epochs=100, verbose=1).history["loss"][0]
    prediction = model.predict(np.array(X_test.drop(columns=["name", "game_date"])))
    r2 = r2_score(y_test, prediction)
    print(r2)

    return X_test.assign(predicted_points=prediction, actual_points=y_test)


def _pick_best_team(game_date: pd.DataFrame) -> pd.DataFrame:
    predicted_best_team = np.sum(game_date.sort_values("predicted_points", ascending=False)["predicted_points"][:9])
    actual_team_picked = np.sum(game_date.sort_values("predicted_points", ascending=False)["actual_points"][:9])
    actual_best_team = np.sum(game_date.sort_values("actual_points", ascending=False)["actual_points"][:9])
    return (actual_team_picked, actual_best_team)


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