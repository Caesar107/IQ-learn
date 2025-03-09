import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from imitation.data import rollout
from make_envs import make_env

rng=np.random.RandomState(0)

def generate_expert_data(env_name, transition_truncate_len, num_episodes=512, save_path="./experts"):
    class ArgsWrapper:
        def __init__(self, env_name):
            self.env = EnvNameWrapper(env_name)

    class EnvNameWrapper:
        def __init__(self, env_name):
            self.name = env_name

    env_args = ArgsWrapper(args.env)  # 创建兼容 `make_env()` 的对象
    env = make_env(env_args)  # 传入兼容对象


    # 加载 PPO 训练的专家策略
    expert = PPO.load(
    f"{save_path}/{env_name}",
    custom_objects={"observation_space": env.observation_space, "action_space": env.action_space}
)

    # 生成专家轨迹
    trajs = rollout.generate_trajectories(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=num_episodes),
        rng=rng,
    )

    # 确保 `rewards` 被存入
    for traj in trajs:
        if "reward" in traj["infos"][0]:
            traj["rewards"] = np.array([info["reward"] for info in traj["infos"]])
        else:
            traj["rewards"] = np.zeros(len(traj["obs"]))  # 没有 `reward`，填充 0
            print(f"⚠️ 警告: 轨迹中没有 `reward` 数据，已填充 0")

    # Flatten 轨迹
    transitions = rollout.flatten_trajectories(trajs)
    transitions = transitions[:transition_truncate_len]  # 截断数据

    # 保存数据
    save_file = f"{save_path}/transitions_{env_name}.npy"
    torch.save(transitions, save_file)
    print(f"✅ 已保存专家数据到 {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym 环境名称 (如 CartPole-v1)")
    parser.add_argument("--transition_truncate_len", type=int, default=256, help="截断 `transitions` 长度")
    args = parser.parse_args()

    generate_expert_data(args.env, args.transition_truncate_len)
