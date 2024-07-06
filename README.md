# Q-Learning for Noisy CartPole Environment

This project implements a Q-Learning agent to solve the Cart-Pole problem with noisy observations using the Gymnasium library. The project includes a training phase where the Q-Learning agent learns to balance the pole, and a testing phase where the learned policy is visualized.

## Project Description

The Cart-Pole problem involves balancing a pole hinged on a cart by moving the cart left or right. This implementation introduces Gaussian noise into the observations to simulate real-world sensor inaccuracies. The Q-Learning algorithm is used to learn the optimal policy for balancing the pole in this noisy environment.

## Dependencies

The project requires the following Python libraries:
- `gymnasium`: For the Cart-Pole environment
- `numpy`: For numerical operations
- `matplotlib`: For plotting (if you need to visualize training results)
- `pickle`: For saving and loading the Q-Learning model

You can install these dependencies using pip:
```sh
pip install gymnasium numpy matplotlib
```

## How to Run the Code

### Training the Q-Learning Agent

1. Save the provided code in a file named `q_learning_cartpole.py`.
2. Run the script to train the Q-Learning agent:
```sh
python q_learning_cartpole.py
```
The script will train the Q-Learning agent for 500,000 episodes and save the trained Q-table to a file named `q_learning.pkl`.

### Testing the Q-Learning Agent

The testing phase is included in the same script. After training, the script will load the saved Q-table and run a test episode where the learned policy is visualized.

## Code Explanation

### NoisyObservationWrapper Class
This class wraps the Cart-Pole environment to introduce Gaussian noise into the observations:
```python
class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs
```

### QLearningAgent Class
This class implements the Q-Learning algorithm:
- `__init__`: Initializes the agent with the environment and hyperparameters.
- `discretize`: Discretizes continuous observations into discrete states.
- `choose_action`: Chooses an action based on the epsilon-greedy policy.
- `update_q_table`: Updates the Q-table using the Q-Learning update rule.
- `train`: Trains the Q-Learning agent.
- `test`: Tests the trained Q-Learning agent and visualizes the results.
- `save_model`: Saves the trained Q-table to a file.
- `load_model`: Loads the Q-table from a file.

### Main Script
The main script initializes the environment, trains the Q-Learning agent, and tests the trained policy:
```python
if __name__ == "__main__":
    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=0.1)
    agent = QLearningAgent(noisy_train_env)
    agent.train()
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent.env = noisy_test_env
    agent.test()
    test_env.close()
```

## Notes

- You can adjust the noise level by changing the `noise_std` parameter in the `NoisyObservationWrapper` class.
- The number of training episodes and other hyperparameters can be adjusted in the `QLearningAgent` class.
- The Q-table is saved to `q_learning.pkl` after training and loaded from the same file for testing.

Feel free to experiment with different parameters and observe how the agent's performance changes in the noisy environment.
