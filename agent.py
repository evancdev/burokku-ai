from keras import Model
from keras import Sequential
from keras.layers import Dense # type: ignore
from collections import deque
import numpy as np
import random


class DQNAgent:
    """
    Agent that learns to play Tetris using the quality value function

    Parameters:
    - state_size (int): The size of the input state from the environment
    - buffer_size (int): Size of replay buffer
    - batch_size (int): Size of sampled batch from replay buffer
    - discount (float): Discount factor
    - epsilon (float): Exploration rate
    - epsilon_min (float): Minimum exploration rate
    - epsilon_decay (float): Exploration decay rate
    - epsilon_stop_episode (int): What episode does agent stop decr. epsilon
    - min_replay
    - n_neurons (list(int)): list of number of neurons in each inner layer
    - activations (list): list of activations used in each inner layer
    - loss_fun (obj): loss function object
    - optimizer (obj): optimizer object
    """

<<<<<<< HEAD
    def __init__(self, state_size, buffer_size, batch_size,
                 discount, epsilon, epsilon_min, epsilon_stop_episode,
                 n_neurons, activations, loss_fun, optimizer):
=======
    def __init__(self, state_size, buffer_size, batch_size, 
                discount, epsilon, epsilon_min, epsilon_stop_episode,
                n_neurons, activations, loss_fun, optimizer, replay_start_size):
>>>>>>> cb08e87b500e07b23af9099f6bf95bc73a250ee5

        self.discount = discount
        self.state_size = state_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.mem = deque(maxlen=buffer_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.replay_start_size = replay_start_size

        self.model = self.build_model()

    def build_model(self):
        """Builds a Keras deep neural network model"""

        model = Sequential()
        model.add(Dense(
            self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss_fun, optimizer=self.optimizer)

        return model

    def predict_output(self, state):
        """
        Predicts score output from a given state

         Parameters:
         - state (np.array): The current state of the environment

         Returns:
         - score (int): The expected score from a certain state
        """
        return self.model.predict(state, verbose=0)[0]

    def act(self, state):
        """
        Finds action using epsilon-greedy strategy

        Parameters:
        - state (np.array): The current state of the environment

        Returns:
        - score (int): The expected score from a certain state
        """

        if random.random() <= self.epsilon:
            return self.random_output()
        else:
            # First dim represents batch size of one
            state = np.reshape(state, [1, self.state_size])
            return self.predict_output(state)

    def get_best_state(self, states):
        """
        Out of all states, return the best state, meaning the best piece and rotation to place
        """
        if random.random() <= self.epsilon:
            # If random value is leq exploration variable, choose randomly from available states
            return random.choice(list(states))

        else:
            max_val = float("-inf")
            best_state = None

            for state in states:
                value = self.predict_output(
                    np.reshape(state, [1, self.state_size]))
                if value > max_val:
                    max_val = value
                    best_state = state

        return best_state

    def random_output(self):
        """
        Returns a random score output
        """
        return random.random()

    def remember(self, state, next_state, reward, done):
        """
        Store expereince in replay buffer for training

        Parameters:
        - state (np.array): state of environment
        - action (int, int): Selected action comprising of designated location of piece and rotation
        - reward (int): The reward received after taking the action
        - next_state (np.array): The state of the environment after taking the action
        - done (bool): Whether the episode has ended.
        """
        self.mem.append((state, next_state, reward, done))

<<<<<<< HEAD
    def train(self, epochs=5):
        """
        Samples batch of experiences and train them
        """
        n = len(self.mem)

        ### CHECK IF CONDITION GOOD ###
        if n >= self.batch_size and n >= self.buffer_size:
            batch = random.sample(self.mem, self.batch_size)

            next_states = np.array((x[1] for x in batch))

            # Generate Q-values for every possible next states
            next_q_values = []
            for x in self.model.predict(next_states):
                next_q_values.append(x[0])

            x = []
            y = []

            # batch has parameters (current_state, action, next_state, reward, done)
            for i, (current_state, _, reward, done) in enumerate(batch):
                if done:
                    new_q_value = reward
                else:
                    # reward for taking in a state + discount rate * (max reward from future)
                    new_q_value = reward + self.discount * next_q_values[i]

                x.append(current_state)
                y.append(new_q_value)

            # Fit model
            self.model.fit(np.array(x), np.array(
                y), batch_size=self.batch_size, epochs=epochs, verbose=0)

            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
=======
    def train(self, batch_size=32, epochs=3):
            """Samples batch of experiences and train them"""
            n = len(self.mem)

            if n >= self.replay_start_size and n >= batch_size:

                batch = random.sample(self.mem, batch_size)

                next_states = []
                for x in batch:
                    next_states.append(x[1])
                next_states = np.array(next_states)

                # Generate Q-values for every possible next states
                next_qs = []
                for x in self.model.predict(next_states):
                    next_qs.append(x[0])

                X = []
                Y = []

                # batch has parameters (current_state, action, next_state, reward, done)
                for i, (state, action, reward, done) in enumerate(batch):
                    if not done:
                        new_q_value = reward + self.discount * next_qs[i]
                    else:
                        new_q_value = reward

                    X.append(state)
                    Y.append(new_q_value)

                # Fit model
                self.model.fit(np.array(X), np.array(Y), batch_size = batch_size, epochs = epochs, verbose = 0)

                # Update epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay
                
>>>>>>> cb08e87b500e07b23af9099f6bf95bc73a250ee5
