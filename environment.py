# environment.py

import gymnasium as gym
import numpy as np

def setup_environment():
    """
    Sets up the CartPole-v1 environment and runs a random policy to test it.
    """
    # Initialize the CartPole-v1 environment
    env = gym.make('CartPole-v1')
    
    # Run a few episodes with a random policy
    for episode in range(5):
        observation, info = env.reset()
        done = False
        while not done:
            env.render()
            action = env.action_space.sample()  # Random action
            observation, reward, done, _, _ = env.step(action)
            print(f"Episode {episode + 1} | Observation: {observation} | Reward: {reward} | Done: {done}")
    
    env.close()

if __name__ == "__main__":
    setup_environment()
