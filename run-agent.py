from agent import DQNAgent
from tetris import Tetris
from heuristic import TetrisAI
from tqdm import tqdm
import numpy as np


class AgentParam:
    def __init__(self):
        self.discount = 0.95
        self.state_size = 4
        self.batch_size = 512
        self.buffer_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.n_neurons = [32, 32]
        self.activations = ['relu', 'relu', 'linear']
        self.loss_fun = 'mse'
        self.optimizer = 'adam'
        self.epsilon_stop_episode = 2000
        self.max_steps = 1000
        self.episodes = 2000
        self.epochs = 1
        self.replay_start_size = 2000


def run_dqn(agentparam: AgentParam):
    tetris = Tetris()
    scores = []

    agent = DQNAgent(
        discount=agentparam.discount,
        state_size=agentparam.state_size,
        batch_size=agentparam.batch_size,
        buffer_size=agentparam.buffer_size,
        epsilon=agentparam.epsilon,
        epsilon_min=agentparam.epsilon_min,
        n_neurons=agentparam.n_neurons,
        activations=agentparam.activations,
        loss_fun=agentparam.loss_fun,
        optimizer=agentparam.optimizer,
        epsilon_stop_episode=agentparam.epsilon_stop_episode,
        replay_start_size=agentparam.replay_start_size
    )

    for episode in tqdm(range(agentparam.episodes)):
        curr_state = tetris.reset()
        done = False
        step = 0

        print(f"Episode {episode + 1}/{agentparam.episodes}")
        
        while not done and (agentparam.max_steps != 0 or step < agentparam.max_steps):
            next_states = tetris.get_next_states()

            if not next_states:
                print("No next states available. Ending episode.")
                break

            best_state = agent.get_best_state(list(next_states.keys()))
            if best_state is None:
                print("Agent failed to select a valid state.")
                break

            tetris.rotate_piece(best_state[1])
            reward, done = tetris.play(best_state[0], render=True)
            
            agent.remember(curr_state, best_state, reward, done)
            curr_state = best_state
            step += 1

        print(f"Training on episode {episode + 1}")
        scores.append(tetris.get_score())
        agent.train(batch_size=agentparam.batch_size, epochs=agentparam.epochs)
    
    print("scores", scores)






if __name__ == "__main__":
    agentparam = AgentParam()
    run_dqn(agentparam)
