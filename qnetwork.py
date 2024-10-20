import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNN(nn.Module):
  class QNetwork(nn.Module):
    """
    QNN Network
    
    Param:
    - state_size (int): The size of the input, which represents the current state.
    - action_size (int): The number of possible actions the agent can take.

    Methods:
    - forward(state): Forward pass to predict Q-values for the given state.
    """

  def __init__(self, state_size, action_size):
    pass


  def forward(self, state):
      """
      Performs the forward pass through the QNN to predict Q-values given state

      Parameters:
      - state (torch.Tensor (?)): Input state of environment

      Returns:
      - Q-values (torch.Tensor): A tensor of Q-values for each action in possible actions
      """
