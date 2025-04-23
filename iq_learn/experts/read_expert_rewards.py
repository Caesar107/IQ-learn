import argparse
import gym
import numpy as np
from stable_baselines3 import PPO


def extract_rewards(model_zip: str, env_id: str, episodes: int):
    """
    Load an SB3 policy from a zip file and run it in the environment to collect episode rewards.

    :param model_zip: Path to the SB3 model zip (e.g. Ant-v2.zip)
    :param env_id: Gym environment ID (e.g. 'Ant-v2')
    :param episodes: Number of episodes to run
    :return: NumPy array of episode rewards
    """
    # Load the expert policy from the zip
    model = PPO.load(model_zip)
    # Create the environment
    env = gym.make(env_id)

    rewards = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {ep+1}/{episodes}: reward = {total_reward}")

    env.close()
    return np.array(rewards)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save episode rewards from an expert model zip."
    )
    parser.add_argument(
        "model_zip",
        help="Path to the SB3 model zip (e.g. experts/expert_data/Ant-v2.zip)"
    )
    parser.add_argument(
        "--env", default="Ant-v2",
        help="Gym environment ID (default: Ant-v2)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes to roll out (default: 100)"
    )
    parser.add_argument(
        "-o", "--output", default="ant_v2_episode_rewards.npy",
        help="Output .npy file for episode rewards"
    )
    args = parser.parse_args()

    rewards = extract_rewards(args.model_zip, args.env, args.episodes)
    # Save rewards to disk
    np.save(args.output, rewards)
    print(f"Saved {len(rewards)} episode rewards to {args.output}")
    print(f"Mean reward: {rewards.mean():.2f}, Std: {rewards.std():.2f}")


if __name__ == "__main__":
    main()
