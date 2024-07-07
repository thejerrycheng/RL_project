import gymnasium as gym
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
import argparse
import datetime

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2,  # cart position range is [-2.4, 2.4]
            0.5,  # cart velocity range is approximated as [-50, 50]
            math.radians(20),  # pole angle range is [-12 degrees, 12 degrees]
            math.radians(0.5)  # pole angular velocity range is approximated as [-50 degrees/s, 50 degrees/s]
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs

    # def __init__(self, env, noise_std=0):
    #     super(NoisyObservationWrapper, self).__init__(env)
    #     self.noise_std = noise_std

    # def observation(self, obs):
    #     noise = np.random.normal(0.1, self.noise_std, size=obs.shape)
    #     noisy_obs = obs + noise
    #     return noisy_obs

class QLearningAgent:
    def __init__(self, env, bins=(50, 50, 50, 50), alpha=0.1, gamma=0.7, epsilon=0.7, epsilon_decay=0.999, epsilon_min=0.01, model_filename=None):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros(self.bins + (env.action_space.n,))
        
        # Define smaller bin limits compared to the terminal states
        self.bins_limits = [
            (-2.4, 2.4),  # cart position (terminal state is -4.8 to 4.8)
            (-10, 10),  # cart velocity (terminal state is -inf to inf, but we use a practical range)
            (-math.radians(12), math.radians(12)),  # pole angle (terminal state is -math.radians(24) to math.radians(24))
            (-math.radians(10), math.radians(10))  # pole angular velocity (terminal state is -inf to inf, but we use a practical range)
        ]
        
        # Load the model if specified
        if model_filename:
            self.load_model(model_filename)
 
    def discretize(self, obs):
        # Discretize the continuous observations
        ratios = [(obs[i] - self.bins_limits[i][0]) / (self.bins_limits[i][1] - self.bins_limits[i][0]) for i in range(len(obs))]
        new_obs = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.bins[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state):
        if np.random.random() < 0.3:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action] * (not done)
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def train(self, save_filename):
        best_total_reward = -float('inf')  # Initialize the best total reward
        rewards = []  # Initialize list to store rewards for each episode

        episode = 0
        for epiosde in range(1000000): #5,000,000
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
            rewards.append(total_reward)  # Store the total reward for this episode

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.save_model(filename=save_filename)  # Save the model if the current total reward is higher than the best total reward
                print(f"Episode: {episode}, Best total reward: {best_total_reward} ") # Print the best total reward

            if episode % 100 == 0:
                print(f"Episode: {episode}, Total reward: {total_reward}")

            episode += 1

        # Plot episode vs. rewards
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs. Total Reward')
        plt.show()

    def test(self, episodes=20):
        total_rewards = []

        for episode in range(episodes):
            current_state, info = self.env.reset()
            current_state = self.discretize(current_state)
            done = False
            total_reward = 0

            while not done:
                if episode % 20 == 0:
                    self.env.render()
                action = np.argmax(self.q_table[current_state])  # Choose the action with the highest Q-value
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize(next_state)
                current_state = next_state
                total_reward += reward

            total_rewards.append(total_reward)
            print(f"Test Episode: {episode}, Total reward: {total_reward}")

        average_reward = np.mean(total_rewards)
        highest_reward = np.max(total_rewards)
        lowest_reward = np.min(total_rewards)

        print(f"Average Reward: {average_reward}")
        print(f"Highest Reward: {highest_reward}")
        print(f"Lowest Reward: {lowest_reward}")

        self.env.close()


    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Q-Learning for CartPole with Sensor Noise')
    parser.add_argument('--load', type=str, help='Model filename to load', default=None)
    parser.add_argument('--save', type=str, help='Model filename to save', default=None)
    args = parser.parse_args()

    save_filename = args.save
    if not save_filename:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f'q_learning_{current_time}.pkl'

    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=0.1)
    agent = QLearningAgent(noisy_train_env, model_filename=args.load)
    agent.train(save_filename)
    
    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent.env = noisy_test_env
    agent.test()
    test_env.close()