# Burokku-AI

A Tetris AI implementation using Deep Q-Network (DQN) reinforcement learning with comparison to heuristic-based approaches.

## Overview

This project implements an AI agent that learns to play Tetris using deep reinforcement learning. The implementation includes:

- **Custom Tetris Environment**: A fully functional Tetris game with standard rules and scoring
- **DQN Agent**: Deep Q-Network agent using neural networks to learn optimal play strategies
- **Heuristic AI**: Traditional heuristic-based AI for performance comparison
- **State Representation**: Board analysis using aggregate height, holes, bumpiness, and cleared lines

## Implementation Details

### Architecture

#### Tetris Environment ([tetris.py](tetris.py))
- Board size: 10x20 grid
- Standard 7 tetromino pieces (I, O, T, S, Z, J, L)
- State features: aggregate height, holes, bumpiness, cleared lines
- Rendering support using OpenCV
- Reward system based on cleared lines and game progression

#### DQN Agent ([agent.py](agent.py))
- **State size**: 4 features (aggregate height, holes, bumpiness, cleared lines)
- **Network architecture**: Configurable neural network with dense layers
  - Default: 2 hidden layers with 32 neurons each
  - Activations: ReLU for hidden layers, linear for output
- **Experience replay**: Deque-based replay buffer (default: 1000 samples)
- **Exploration**: Epsilon-greedy policy with decay
  - Initial epsilon: 1.0
  - Minimum epsilon: 0.01
  - Decay over 1500 episodes
- **Training**: Batch learning from replay buffer (batch size: 512)
- **Optimizer**: Adam optimizer with MSE loss

#### Heuristic AI ([heuristic.py](heuristic.py))
- Evaluates all possible piece placements using weighted heuristic function
- Weights:
  - Aggregate height: -0.5
  - Holes: -0.7
  - Bumpiness: -0.3
  - Cleared lines: +1.0
- Used for baseline comparison with DQN agent

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### Key Dependencies

- **tensorflow**: Neural network framework for DQN
- **keras**: High-level API for building neural networks
- **numpy**: Numerical computations and array operations
- **opencv-python**: Game rendering and visualization
- **torch**: Additional ML framework support
- **tqdm**: Progress bars for training

## How to Run

### Training the DQN Agent

Run the training script to train a new DQN agent:

```bash
python run-agent.py
```

Training parameters (configurable in [run-agent.py](run-agent.py)):
- Episodes: 2000
- Max steps per episode: 1000
- Discount factor: 0.98
- Batch size: 512
- Buffer size: 1000

The script will save the best performing model as `best_model.keras` when it reaches the target score.

### Testing a Trained Model

To watch a trained agent play Tetris:

```bash
python test_model.py
```

This will load `best_model.keras` and render the agent playing in real-time with OpenCV visualization.

### Running Heuristic AI

Test the heuristic-based AI approach:

```bash
python tetris-test.py
```

This demonstrates the traditional heuristic search algorithm for comparison with the learned DQN agent.

## Project Structure

```
Burokku-AI/
├── agent.py           # DQN agent implementation
├── tetris.py          # Tetris environment and game logic
├── heuristic.py       # Heuristic-based AI
├── run-agent.py       # Training script for DQN agent
├── test_model.py      # Testing script for trained models
├── tetris-test.py     # Heuristic AI testing
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Model Performance

The DQN agent is trained to maximize score by:
- Learning to clear multiple lines simultaneously
- Minimizing board holes and height
- Maintaining flat board surface (low bumpiness)

Training progress is displayed with score tracking and best model saving when performance thresholds are exceeded.