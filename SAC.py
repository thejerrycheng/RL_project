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
import csv

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        
        # Define the range for each observation component
        self.obs_ranges = [
            2.4,  # cart position noise range is -2 to 2
            0.5,  # cart velocity noise range is -0.5 to 0.5
            math.radians(12),  # pole angle noise range is -20 degrees to 20 degrees
            math.radians(0.5)  # pole angular velocity noise range is -0.5 degrees/s to 0.5 degrees/s
        ]

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape) * self.obs_ranges
        noisy_obs = obs + noise
        return noisy_obs

class DiscreteToContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscreteToContinuousWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(env.action_space.n,), dtype=np.float32
        )

    def action(self, action):
        action = np.argmax(action)
        return action

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.log_std = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc3(x))
        log_std = self.log_std(x).clamp(-20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(dim=-1)
        action = torch.tanh(action)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SACAgent:
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2, lr=0.0003, batch_size=64, memory_size=100000):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.critic1 = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(self.device)
        self.critic2 = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(self.device)
        self.target_critic1 = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(self.device)
        self.target_critic2 = Critic(env.observation_space.shape[0] + env.action_space.shape[0], 1).to(self.device)
        self.update_target_networks(1.0)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.memory = deque(maxlen=memory_size)
        self.loss_fn = nn.MSELoss()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()

    def sample_experience(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def train(self, episodes=1000, save_filename=None):
        if save_filename is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f'sac_{current_time}.pth'

        rewards = []
        best_total_reward = -float('inf')
        for episode in range(episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step += 1
                if len(self.memory) > self.batch_size:
                    self.update()

            rewards.append(episode_reward)

            if episode_reward > best_total_reward:
                best_total_reward = episode_reward
                self.save_model(save_filename)
                print(f"New best total reward: {best_total_reward} - Model saved")

            if episode % 10 == 0:
                print(f"Episode: {episode}, Reward: {episode_reward}, Steps: {step}")

            if step >= 500:
                print("SUCCESS!")
                break

        # Save rewards to CSV file
        rewards_filename = save_filename.replace('.pth', '_rewards.csv')
        with open(rewards_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total Reward'])
            for i, reward in enumerate(rewards):
                writer.writerow([i, reward])

        # Plot rewards
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode vs. Reward')
        plt.savefig(save_filename.replace('.pth', '_plot.png'))
        plt.show()

    def update(self):
        states, actions, rewards, next_states, dones = self.sample_experience()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], 1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], 1))
            target_q = rewards + (1 - dones) * self.gamma * (torch.min(target_q1, target_q2) - self.alpha * next_log_probs)

        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))

        critic1_loss = self.loss_fn(current_q1, target_q)
        critic2_loss = self.loss_fn(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(torch.cat([states, new_actions], 1))
        q2 = self.critic2(torch.cat([states, new_actions], 1))
        actor_loss = (self.alpha * log_probs - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()

    def save_model(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict()
        }, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        print(f"Model loaded from {filename}")

    def test(self, model_filename, episodes=10):
        self.load_model(model_filename)
        for episode in range(episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            while not done:
                if args.render:
                    self.env.render()
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                step += 1
                if step >= 500:
                    print("SUCCESS!")
                    break
            print(f"Test Episode: {episode}, Reward: {episode_reward}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC for CartPole with Sensor Noise')
    parser.add_argument('--load', type=str, help='Model filename to load', default=None)
    parser.add_argument('--save', type=str, help='Model filename to save', default=None)
    parser.add_argument('--update_rate', type=int, default=10, help='Target network update rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy regularization coefficient')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for replay')
    parser.add_argument('--memory_size', type=int, default=100000, help='Replay memory size')
    parser.add_argument('--noise_std', type=float, default=0.1, help='Standard deviation of the noise')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    save_filename = args.save
    if not save_filename:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f'sac_{current_time}.pth'

    def create_env(render_mode=None, noise_std=args.noise_std):
        env = gym.make('CartPole-v1', render_mode=render_mode)
        env = DiscreteToContinuousWrapper(env)
        return NoisyObservationWrapper(env, noise_std=noise_std)

    if args.test:
        test_env = create_env(render_mode='human' if args.render else None, noise_std=args.noise_std)
        agent = SACAgent(test_env, gamma=args.gamma, tau=args.tau, alpha=args.alpha, lr=args.lr, batch_size=args.batch_size, memory_size=args.memory_size)
        if args.load:
            agent.load_model(args.load)
        agent.test(model_filename=args.load, episodes=10)
    else:
        train_env = create_env(noise_std=args.noise_std)
        agent = SACAgent(train_env, gamma=args.gamma, tau=args.tau, alpha=args.alpha, lr=args.lr, batch_size=args.batch_size, memory_size=args.memory_size)
        if args.load:
            agent.load_model(args.load)
        agent.train(episodes=args.episodes, save_filename=save_filename)

        test_env = create_env(render_mode='human' if args.render else None, noise_std=args.noise_std)
        agent.env = test_env
        agent.test(model_filename=save_filename, episodes=10)
        test_env.close()
