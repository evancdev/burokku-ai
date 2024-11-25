from agent import DQNAgent
from tetris import Tetris
from heuristic import TetrisAI
from tqdm import tqdm


class AgentParam:
    def __init__(self):
        self.discount = 0.9
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
        epsilon_stop_episode=agentparam.epsilon_stop_episode
    )

    for episode in tqdm(range(agentparam.episodes)):
        curr_state = tetris.reset()
        done = False
        step = 0

        while not done and (agentparam.max_steps == 0 or step < agentparam.max_steps):
            print(tetris.get_piece())
            print(tetris.board)
            print(tetris.curr_piece)
            next_states = tetris.get_next_states()
            print("next_states", next_states)
            best_state = agent.get_best_state(list(next_states.values()))
            print("best_state", best_state)

            # Find the action that leads to the best state
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
            print("BEST ACTION", best_action)
            print("0TH INDEX", best_action[0])
            tetris.rotate_piece(best_action[1])
            reward, done = tetris.play(best_action[0])
            agent.remember(curr_state, next_states[best_action], reward, done)
            curr_state = next_states[best_action]
            step += 1
            agent.train()


if __name__ == "__main__":
    agentparam = AgentParam()
    run_dqn(agentparam)
