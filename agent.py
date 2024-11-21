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

    Methods:
    - act(state): Use epsilon-greedy policy to play action based on the current state
    - remember(state, action, reward, next_state, done): Store experience in replay buffer
    - train(): Sample batch of experiences and train in replay buffer while updating QNN
    """

    def __init__(self, state_size,
                  buffer_size, batch_size, gamma, 
                  learning_rate, epsilon, 
                  epsilon_min, epsilon_decay):
      
      self.discount = gamma
      self.state_size = state_size
      self.batch_size = batch_size
      self.buffer_size = buffer_size
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.learning_rate = learning_rate

      self.model = self.build_model()
    
    def build_model(self):
       """
       Builds a Keras dense neural network
       """


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


    def train(self):
        """
        Samples batch of experiences and train them

        """
        pass