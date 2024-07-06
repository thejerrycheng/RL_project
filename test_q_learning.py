# test_q_learning_cartpole.py

import gymnasium as gym
import numpy as np
import math
import pickle

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs

class QLearningAgent:
    def __init__(self, env, bins=(6, 12, 6, 12), alpha=0.1, gamma=0.7, epsilon=0.7, epsilon_decay=0.9, epsilon_min=0.01):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros(self.bins + (env.action_space.n,))
        
        # Define the bin limits
        self.bins_limits = [
            (-2.4, 2.4),  # cart position
            (-10, 10),  # cart velocity
            (-math.radians(12), math.radians(12)),  # pole angle
            (-math.radians(50), math.radians(50))  # pole angular velocity
        ]

    def discretize(self, obs):
        ratios = [(obs[i] - self.bins_limits[i][0]) / (self.bins_limits[i][1] - self.bins_limits[i][0]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filename}")

    def test(self, episodes=1):
        self.load_model('q_learning.pkl')
        for episode in range(episodes):
            current_state, info = self.env.reset()
            current_state = self.discretize(current_state)
            done = False
            total_reward = 0

            while not done:
                self.env.render()
                action = np.argmax(self.q_table[current_state])
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(next_state)
                current_state = next_state
                total_reward += reward

            print(f"Test Episode: {episode}, Total reward: {total_reward}")

if __name__ == "__main__":
    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent = QLearningAgent(noisy_test_env)
    agent.test()
    test_env.close()
