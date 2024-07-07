

# RL_PROJECT

This project involves implementing reinforcement learning algorithms to solve the Cart-Pole problem. The goal is to develop and compare traditional non-deep RL algorithms and deep RL algorithms to evaluate their performance under sensor noise.

## Description of Scripts

### Training Scripts

- `DQN.py`: This script contains the implementation of the Deep Q-Network (DQN) algorithm. It trains the DQN agent on the Cart-Pole environment.

- `new_q_learning.py`: This script trains the Q-Learning agent on the Cart-Pole environment with noise added to the observations.

- `q_learning_cartpole.py`: This script contains the implementation of the Q-Learning algorithm. It trains the Q-Learning agent on the Cart-Pole environment.

### Testing Scripts

- `test_dqn.py`: This script tests the trained DQN model on the Cart-Pole environment. It loads a pre-trained DQN model and runs a test episode to visualize the agent's performance.

- `test_q_learning.py`: This script tests the trained Q-Learning model on the Cart-Pole environment. It loads a pre-trained Q-Learning model and runs a test episode to visualize the agent's performance.

### Environment Scripts

- `environment.py`: This script sets up the basic Cart-Pole environment using Gymnasium.

- `environment_with_noise.py`: This script modifies the Cart-Pole environment to introduce noise in its sensors, simulating inaccuracies in position, velocity, pole angle, and angular velocity.

## How to Run

### Training Q-Learning Agent

To train the Q-Learning agent, run the following command:

```bash
python q_learning_cartpole.py
```

### Training DQN Agent

To train the DQN agent, run the following command:

```bash
python DQN.py
```

### Testing Q-Learning Agent

To test the Q-Learning agent, run the following command:

```bash
python test_q_learning.py
```

### Testing DQN Agent

To test the DQN agent, run the following command:

```bash
python test_dqn.py
```

## Model and Data Files

### .pkl Files

The `.pkl` files are used to save the trained models and Q-tables. Here are some of the key files:

- `dqn_0.1_new.pkl`, `dqn_0.1.pkl`, `dqn_best_model.pkl`: These files store the trained DQN models.
- `q_learning_0.pkl`, `q_learning_2.pkl`, `q_learning_3.pkl`, `q_learning_best.pkl`, etc.: These files store the trained Q-Learning models.
- `qlearning_Q.pkl`: This file stores the Q-table for the Q-Learning agent.

### Comments on Generated Files

- The `q_learning_best.pkl` file contains the best-performing Q-Learning model based on the total reward.
- The `dqn_best_model.pkl` file contains the best-performing DQN model.
- The various other `.pkl` files represent different stages of training, noise configurations, or parameter settings.

## Notes

- Ensure all dependencies are installed before running the scripts. Key dependencies include Gymnasium, NumPy, Matplotlib, and any other libraries specified in the scripts.
- The training scripts may take a significant amount of time to run, depending on the number of episodes and the complexity of the model.
```

You can create a file named `README.md` and paste this content into it.