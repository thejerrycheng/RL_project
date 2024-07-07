# dqn_cartpole.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import datetime
import argparse
import matplotlib.pyplot as plt

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, memory_size=10000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def train(self, episodes=5000, save_filename=None):
        if save_filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f'dqn_{current_time}.pth'

        rewards = []
        recent_rewards = deque(maxlen=10)
        best_total_reward = -float('inf')

        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                # Custom reward function
                cart_position, cart_velocity, pole_angle, pole_velocity = next_state
                reward = (1.0 - (abs(cart_position) / 4.8) - (abs(pole_angle) / 0.418))
                
                if done and total_reward < 500:
                    reward = -1.0 - (abs(cart_position) / 4.8) - (abs(pole_angle) / 0.418) # Penalize if the episode ends prematurely

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                self.replay()

            rewards.append(total_reward)
            recent_rewards.append(total_reward)

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.save_model(save_filename)
                print(f"New best total reward: {best_total_reward} - Model saved")

            if episode % 10 == 0:
                self.update_target_model()

            if episode % 100 == 0:
                print(f"Episode: {episode}, Total reward: {total_reward}, Average reward: {np.mean(rewards[-100:])}, Epsilon: {self.epsilon}")

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs. Total Reward')
        plt.show()

    def test(self, model_filename, episodes=10):
        self.load_model(model_filename)
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                self.env.render()
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action = np.argmax(self.model(state).cpu().data.numpy())
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                total_reward += reward

            print(f"Test Episode: {episode}, Total reward: {total_reward}")

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN for CartPole with Sensor Noise')
    parser.add_argument('--load', type=str, help='Model filename to load', default=None)
    parser.add_argument('--save', type=str, help='Model filename to save', default=None)
    args = parser.parse_args()

    save_filename = args.save
    if not save_filename:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f'dqn_{current_time}.pth'

    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=0.1)
    agent = DQNAgent(noisy_train_env)
    if args.load:
        agent.load_model(args.load)
    agent.train(save_filename=save_filename)

    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent.env = noisy_test_env
    agent.test(model_filename=save_filename, episodes=10)
    test_env.close()
