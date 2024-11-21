from keras import Model
from keras import Sequential
from tensorflow.python.keras.layers import Dense
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
    - gamma (float): Discount factor
    - learning_rate (float): Learning rate
    - epsilon (float): Exploration rate
    - epsilon_min (float): Minimum exploration rate
    - epsilon_decay (float): Exploration decay rate
    - n_neurons (list(int)): list of number of neurons in each inner layer
    - activations (list): list of activations used in each inner layer
    - loss_fun (obj): loss function object
    - optimizer (obj): optimizer object

    Methods:
    - act(state): Use epsilon-greedy policy to play action based on the current state
    - remember(state, action, reward, next_state, done): Store experience in replay buffer
    - train(): Sample batch of experiences and train in replay buffer while updating QNN
    """

    def __init__(self, state_size,
                 buffer_size, batch_size, gamma,
                 learning_rate, epsilon,
                 epsilon_min, epsilon_decay, n_neurons, activations, loss_fun, optimizer):

        self.discount = gamma
        self.state_size = state_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.mem = deque(maxlen=buffer_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss_fun = loss_fun
        self.optimizer = optimizer

        self.model = self.build_model()

    def build_model(self):
        """
        Builds a Keras dense neural network
        """

        model = Sequential()
        model.add(Dense(
            self.n_neurons[0], input_dimension=self.state_size, activation=self.activations[0]))
        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))
        model.compile(loss=self.loss_fun, optimizer=self.optimizer)

        return model

    def act(self, state):
        """
        Finds action using epsilon-greedy strategy

        Parameters:
        - state (np.array): The current state of the environment

        Returns:
        - action (int, int): Selected action comprising of designated location of piece and rotation
        """

        pass

    def remember(self, state, action, reward, next_state, done):
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

    def train(self):
        """
        Samples batch of experiences and train them

        """
        pass
