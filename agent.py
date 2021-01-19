import random

from collections import deque
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.optimizers import Adam


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """
    Huber loss - Custom Loss Function for Q Learning
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """
    NBA fantasty team bot
    """
    def __init__(self, action_size, state_size: int = 10, strategy="dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy

        # agent config
        self.state_size = state_size
        self.action_size = action_size
        self.model_name = model_name
        self.maxlen = 1000
        self.memory = deque(maxlen=self.maxlen)

        # configure the team
        self.players = list(range(0, self.action_size))
        self.point_guards = []
        self.shooting_guards = []
        self.small_forwards = []
        self.power_forwards = []
        self.centres = []

        # model config
        self.model_name = model_name
        self.gamma = 0.01
        self.epsilon = 1
        self.epsilon_min = 0.25
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.optimizer = Adam(lr=self.learning_rate)

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """
        Creates the model
        """
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=512, activation="tanh"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Adds relevant data to memory
        """
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state, is_eval=False):
        """
        For a given state, pick players from the action space
        """
        if not is_eval and random.uniform(0, 1) <= self.epsilon:
            # self.team.append(random.sample(self.point_guards, 2))
            # self.team.append(random.sample(self.shooting_guards, 2))
            # self.team.append(random.sample(self.small_forwards, 2))
            # self.team.append(random.sample(self.power_forwards, 2))
            # self.team.append(random.sample(self.centres, 1))
            return random.sample(self.players, 9), [1] * 9
        else:
            action_probs = self.model.predict(state)
            # Add a function to optimize a team picking from each position, not just the best 9 players
            # include their salaries and keep below the $60k salary cap
            players = action_probs[0].argsort()[-9:][::-1]
            return (players).tolist(), (action_probs[0][players]).tolist()


    def train_experience_replay(self, batch_size):
        """
        Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)

        X_train, y_train = [], []
        
        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                q_values = self.model.predict(state)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                q_values = self.model.predict(state)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][np.argmax(self.model.predict(next_state)[0])]

                q_values = self.model.predict(state)
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])
                
        else:
            raise NotImplementedError()

        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save(f"models/model_{episode}")

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
