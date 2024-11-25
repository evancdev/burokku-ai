from agent import DQNAgent
from tetris import Tetris
from heuristic import TetrisAI
from tqdm import tqdm


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

        while not done and (agentparam.max_steps != 0 or step < agentparam.max_steps):
            next_states = tetris.get_next_states()

            if next_states == {}:
                break

            best_state = agent.get_best_state(list(next_states.keys()))
            # print("BEST ACTION", best_action)
            # print("0TH INDEX", best_action[0])
            # print("BEFORE")
            # print(tetris.curr_piece)
            tetris.rotate_piece(best_state[1])
            # print("ROTATED")
            # print(tetris.curr_piece)
            reward, done = tetris.play(best_state[0], render=True)
            agent.remember(curr_state, best_state, reward, done)
            curr_state = best_state
            step += 1
        agent.train(batch_size=agentparam.batch_size, epochs=agentparam.epochs)


if __name__ == "__main__":
    agentparam = AgentParam()
    run_dqn(agentparam)
