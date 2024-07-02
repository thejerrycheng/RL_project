# q_learning_cartpole.py

import gymnasium as gym
import numpy as np
import math

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs

class QLearningAgent:
    def __init__(self, env, bins=(6, 12, 6, 12), alpha=0.1, gamma=0.9, epsilon=0.7, epsilon_decay=0.995, epsilon_min=0.01):
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
            (-1, 1),  # cart velocity
            (-math.radians(12), math.radians(12)),  # pole angle
            (-math.radians(50), math.radians(50))  # pole angular velocity
        ]

    def discretize(self, obs):
        ratios = [(obs[i] - self.bins_limits[i][0]) / (self.bins_limits[i][1] - self.bins_limits[i][0]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample() # if the random number is less than epsilon, take a random action
        return np.argmax(self.q_table[state]) # otherwise, take the best action

    def update_q_table(self, current_state, action, reward, next_state, done):
        best_future_q = np.max(self.q_table[next_state]) # get the best Q-value for the next state
        current_q = self.q_table[current_state][action] # get the current Q-value
        self.q_table[current_state][action] = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * best_future_q * (not done)) # update the Q-value

    def train(self, episodes=100000):
        for episode in range(episodes):
            current_state, info = self.env.reset()
            current_state = self.discretize(current_state)
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(current_state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(next_state)
                self.update_q_table(current_state, action, reward, next_state, done)
                current_state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 100 == 0:
                print(f"Episode: {episode}, Total reward: {total_reward}, Epsilon: {self.epsilon}")

    def test(self, episodes=1):
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
    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=0.1)
    agent = QLearningAgent(noisy_train_env)
    agent.train()
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent.env = noisy_test_env
    agent.test()
    test_env.close()
