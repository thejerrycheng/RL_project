# environment_with_noise.py

import gymnasium as gym
import numpy as np

class NoisyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_std=0.1):
        super(NoisyObservationWrapper, self).__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        noisy_obs = obs + noise
        return noisy_obs

def setup_noisy_environment(noise_std=0.1):
    """
    Sets up the CartPole-v1 environment with noisy observations and runs a random policy to test it.
    """
    # Initialize the CartPole-v1 environment with noisy observations
    env = gym.make('CartPole-v1')
    noisy_env = NoisyObservationWrapper(env, noise_std)

    # Run a few episodes with a random policy in the noisy environment
    for episode in range(5):
        observation, info = noisy_env.reset()
        done = False
        while not done:
            noisy_env.render()
            action = noisy_env.action_space.sample()  # Random action
            observation, reward, done, _, _ = noisy_env.step(action)
            print(f"Episode {episode + 1} | Noisy Observation: {observation} | Reward: {reward} | Done: {done}")

    noisy_env.close()

if __name__ == "__main__":
    # You can change the noise_std to study its impact
    setup_noisy_environment(noise_std=0.1)
