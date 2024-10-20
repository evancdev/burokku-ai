class DQNAgent:
    """
    Agent that learns to play Tetris using the quality value function

    Parameters:
    - state_size (int): The size of the input state from the environment
    - action_size (int): Number of actions agent can take
    - buffer_size (int, optional): Size of replay buffer
    - batch_size (int, optional): Size of sampled batch from replay buffer
    - gamma (float, optional): Discount factor
    - learning_rate (float, optional): Learning rate
    - epsilon (float, optional): Exploration rate
    - epsilon_min (float, optional): Minimum exploration rate
    - epsilon_decay (float, optional): Exploration decay rate

    Methods:
    - act(state): Use epsilon-greedy policy to play action based on the current state
    - remember(state, action, reward, next_state, done): Store experience in replay buffer
    - train(): Sample batch of experiences and train in replay buffer while updating QNN
    """

    def __init__(self, state_size,action_size,
                  buffer_size, batch_size, gamma, 
                  learning_rate, epsilon, 
                  epsilon_min, epsilon_decay):
      pass

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