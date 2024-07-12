import gym

# Create the Walker2d environment
env = gym.make('Walker2d-v3')

# Reset the environment to the initial state
observation = env.reset()

# Loop through a few steps with a random policy
for _ in range(1000):
    env.render(mode='human')  # Render the environment
    action = env.action_space.sample()  # Sample a random action
    observation, reward, done, info = env.step(action)  # Take the action

    if done:
        observation = env.reset()  # Reset the environment if done

# Close the environment
env.close()

