import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Hardcoded environment name
env_name = "Humanoid-v2"

# Create the environment
env = gym.make(env_name)
print(f"Created environment: {env_name}")

# Sample transitions from the expert policy
expert = PPO.load(f"./experts/{env_name}")

# Evaluate expert policy
mean_reward, std_reward = evaluate_policy(model=expert, env=env)
print(f"Average reward of the expert: {mean_reward}, {std_reward}.")

env.close()
