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
import math
import json

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2,  # cart position noise range is -2 to 2
            0.5,  # cart velocity noise range is -0.5 to 0.5
            math.radians(2),  # pole angle noise range is -20 degrees to 20 degrees
            math.radians(0.5)  # pole angular velocity noise range is -0.5 degrees/s to 0.5 degrees/s
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs

# A deeper model for better performance
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def reward_fun1(self, cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
        # Reward for staying near the center
        reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
        
        if done and total_reward < 500:
            reward = -0.1 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)  # Penalize if the episode ends prematurely
        return reward
    
def reward_fun2(self, cart_position, cart_velocity, pole_angle, pole_velocity, total_reward, done):
    if abs(cart_position) < 0.5 and abs(pole_angle) < 0.05:
        reward = 1.0
    else:
        # Penalize deviations from the center
        reward = - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209)

    if done and total_reward < 500:
        reward = -1.0  # Penalize if the episode ends prematurely
    return reward
    

# Add more reward functions for experiments 

class DQNAgent:
    def __init__(self, env, update_rate = 10, gamma=0.9, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.999, lr=0.001, batch_size=64, memory_size=10000, reward_fun='reward_fun1'):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.update_rate = update_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_model = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.reward_fun = globals()[reward_fun]

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


    def train(self, episodes=10000, save_filename=None):
        if save_filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f'dqn_{current_time}.pth'

        rewards = []
        recent_rewards = deque(maxlen=100)
        best_total_reward = -float('inf')

        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            step = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Custom reward function
                cart_position, cart_velocity, pole_angle, pole_velocity = next_state
                
                reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
        
                if done and total_reward < 500:
                    reward = -0.1 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)  # Penalize if the episode ends prematurely

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1
                self.replay()

                if step > 500:
                    done = True
                    print("SUCCESS!")
                else:
                    done = False
                    print("FAILURE!")

            rewards.append(total_reward)
            recent_rewards.append(total_reward)

            if total_reward > best_total_reward:
                best_total_reward = total_reward
                self.save_model(save_filename)
                print(f"New best total reward: {best_total_reward} - Model saved")

            if episode % self.update_rate == 0:
                self.update_target_model()
                print(f"Episode: {episode}, Average reward: {np.mean(rewards[-10:])}")

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Save plot
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode vs. Total Reward')
        plt.savefig(save_filename.replace('.pth', '_plot.png'))
        plt.show()



    def test(self, model_filename, episodes=10, disturbance_step=100, disturbance_magnitude=1.0):
        self.load_model(model_filename)
        for episode in range(episodes):
            state, info = self.env.reset()
            total_reward = 0
            done = False
            step = 0

            while not done:
                self.env.render()
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action = np.argmax(self.model(state).cpu().data.numpy())
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Apply disturbance at the specific time step
                if step == disturbance_step:
                    next_state[3] += disturbance_magnitude  # Apply disturbance to pole angular velocity

                # Custom reward function
                cart_position, cart_velocity, pole_angle, pole_velocity = next_state
                reward = 1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)
                
                if done and step < 500:
                    reward = -1.0 - (abs(cart_position) / 2.4) - (abs(pole_angle) / 0.209) - (abs(cart_velocity) / 1.0) - (abs(pole_velocity) / 1.0)  # Penalize if the episode ends prematurely

                state = next_state
                total_reward += reward
                step += 1
                if step >= 500:
                    done = True
                    print("SUCCESS!")
            
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
    parser.add_argument('--reward', type=str, help='Reward function to use', default='reward_fun1')
    args = parser.parse_args()

    save_filename = args.save
    if not save_filename:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f'dqn_{current_time}.pth'

    train_env = gym.make('CartPole-v1')
    noisy_train_env = NoisyObservationWrapper(train_env, noise_std=0.1)
    agent = DQNAgent(noisy_train_env, reward_fun=args.reward)
    if args.load:
        agent.load_model(args.load)
    agent.train(save_filename=save_filename)

    test_env = gym.make('CartPole-v1', render_mode='human')
    noisy_test_env = NoisyObservationWrapper(test_env, noise_std=0.1)
    agent.env = noisy_test_env
    agent.test(model_filename=save_filename, episodes=10)
    test_env.close()
