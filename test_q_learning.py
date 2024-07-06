import gymnasium as gym
import numpy as np
import math
import pickle
import argparse

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs

class QLearningAgent:
    def __init__(self, env, bins=(16, 16, 16, 16), alpha=0.1, gamma=0.7, epsilon=0.7, epsilon_decay=0.9, epsilon_min=0.01):
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
            (-4.8, 4.8),  # cart position
            (-100, 100),  # cart velocity
            (-math.radians(24), math.radians(24)),  # pole angle
            (-math.radians(100), math.radians(100))  # pole angular velocity
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

    def test(self, model_filename, episodes=1):
        self.load_model(model_filename)
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
    parser = argparse.ArgumentParser(description='Test Q-Learning for CartPole with Sensor Noise')
    parser.add_argument('--model', type=str, required=True, help='Path to the model file to load')
    parser.add_argument('--episodes', type=int, default=1, help='Number of test episodes')
    args = parser.parse_args()

    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent = QLearningAgent(noisy_test_env)
    agent.test(model_filename=args.model, episodes=args.episodes)
    test_env.close()
