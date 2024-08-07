

# RL_PROJECT

This project involves implementing reinforcement learning algorithms to solve the Cart-Pole problem. The goal is to develop and compare traditional non-deep RL algorithms and deep RL algorithms to evaluate their performance under sensor noise.


# QLearning.py: 

#### Overview
This script trains and tests a Q-Learning agent on the CartPole-v1 environment with sensor noise. It supports training, testing, and disturbance testing.

#### How to Run

**Train the Model:**
```sh
python script.py --save model.pkl --bins 16,16,16,16 --noise_std 0.1
```

**Test the Model:**
```sh
python script.py --load model.pkl --test --render
```


# DQN.py: 

#### Overview
This script trains and tests a Deep Q-Network (DQN) agent on the CartPole-v1 environment with sensor noise. It supports training, testing, and disturbance testing.

#### How to Run

**Train the Model:**
```sh
python script.py --save model.pth --episodes 5000 --noise_std 0.1
```

**Test the Model:**
```sh
python script.py --load model.pth --test --render
```

# PPO.py

#### Overview
This script trains and tests a model-based reinforcement learning agent using Proximal Policy Optimization (PPO) for the CartPole-v1 environment with sensor noise. The training process includes collecting data, training a dynamics model, and training a policy network.

#### How to Run

**Collect Data, Train Dynamics Model, and Train Policy:**
```sh
python script.py --num_episodes 5000 --training_iterations 1000 --noise_std 0.1 --postfix my_experiment
```

**Test a Pre-Trained Policy:**
```sh
python script.py --load saved_policy_models/ppo_final_my_experiment.pth
```


# SAC.py

#### Overview
This script trains and tests a Soft Actor-Critic (SAC) agent on the CartPole-v1 environment with sensor noise. It supports training, testing, and disturbance testing.

#### How to Run

**Train the Model:**
```sh
python script.py --save model.pth --episodes 5000 --noise_std 0.1
```

**Test the Model:**
```sh
python script.py --load model.pth --test --render
```

