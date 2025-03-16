import os
import zipfile
import torch
import torch.nn.functional as F
import gym
import numpy as np
from stable_baselines3 import PPO

def load_ppo_from_zip(model_zip_path, env, extract_dir="temp_model"):
    """
    从 zip 文件中手动解压并加载 PPO 模型的策略参数（保存在 policy.pth 中），
    然后构造一个 PPO 模型并加载参数。

    Args:
        model_zip_path (str): 模型 zip 文件路径。
        env: 用于构造 PPO 模型的环境（例如 gym.make("CartPole-v1")）。
        extract_dir (str): 用于临时解压的文件夹名称。

    Returns:
        PPO 模型对象，如果加载失败则返回 None。
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("解压后的文件列表：", os.listdir(extract_dir))
    
    policy_file = os.path.join(extract_dir, "policy.pth")
    if not os.path.exists(policy_file):
        print("未找到 policy.pth 文件，请检查解压后的内容。")
        return None
    
    try:
        policy_state_dict = torch.load(policy_file, map_location="cpu")
        print("加载的 policy.pth 中的 keys:", policy_state_dict.keys())
    except Exception as e:
        print("加载 policy.pth 时发生错误：", e)
        return None
    
    model = PPO("MlpPolicy", env, verbose=1)
    try:
        model.policy.load_state_dict(policy_state_dict)
        print("成功加载 policy.pth 到 PPO 模型中。")
    except Exception as e:
        print("加载 state_dict 时发生错误：", e)
        return None
    
    return model

def evaluate_actions_distribution(model, states):
    """
    通过调用 model.policy.evaluate_actions 方法，计算模型在给定状态下的动作概率分布。
    对于离散动作环境（例如 CartPole），遍历每个可能的动作，
    调用 evaluate_actions 得到 log_prob，然后通过 exp 得到概率，堆叠得到完整分布。
    如果返回的概率张量最后一个维度大于 1，则取该维度的均值。

    Args:
        model: PPO 模型（无论专家或 agent）。
        states: Tensor，形状 [batch_size, state_dim]。

    Returns:
        Tensor，形状 [batch_size, num_actions]，每个元素 >= 1e-7。
    """
    with torch.no_grad():
        n_actions = model.action_space.n
        batch_size = states.shape[0]
        probs_list = []
        print("使用 evaluate_actions 接口获取概率分布:")
        for a in range(n_actions):
            # 构造所有样本都选动作 a 的动作张量
            actions = torch.full((batch_size, 1), a, dtype=torch.int64, device=states.device)
            # evaluate_actions 返回 (values, log_prob, entropy)
            _, log_prob, _ = model.policy.evaluate_actions(states, actions)
            # 这里 log_prob 可能的形状为 [batch_size, 1, X]
            prob = torch.exp(log_prob).squeeze(1)  # 试图 squeeze 第2维
            print(f"  动作 {a} 的原始概率形状: {torch.exp(log_prob).shape}")
            probs_list.append(prob)
        probs = torch.stack(probs_list, dim=1)
        print("evaluate_actions_distribution 返回的原始概率形状:", probs.shape)
        # 如果最后一维不是1，则认为存在冗余信息，取均值
        if probs.dim() == 3 and probs.shape[-1] != 1:
            probs = probs.mean(dim=-1)
            print("取均值后，概率形状:", probs.shape)
        return probs.clamp(min=1e-7)

def get_expert_action_probs(expert_policy, states):
    """
    获取专家策略在给定状态下的动作概率分布。
    优先尝试调用 expert_policy.policy.get_distribution；如果没有，则尝试使用 evaluate_actions_distribution，
    如果仍然没有，则退回使用 expert_policy.predict 构造 one-hot 分布（仅用于调试）。

    Args:
        expert_policy: 专家策略模型（例如 PPO 模型）。
        states: Tensor，形状 [batch_size, state_dim]。

    Returns:
        Tensor，形状 [batch_size, num_actions]。
    """
    with torch.no_grad():
        if hasattr(expert_policy.policy, "get_distribution"):
            try:
                dist = expert_policy.policy.get_distribution(states)
                print("专家模型使用 get_distribution 获取概率分布")
                return dist.distribution.probs.clamp(min=1e-7)
            except Exception as e:
                print("调用专家模型的 get_distribution 出错：", e)
        if hasattr(expert_policy.policy, "evaluate_actions"):
            try:
                print("专家模型使用 evaluate_actions_distribution 获取概率分布")
                probs = evaluate_actions_distribution(expert_policy, states)
                print("专家模型 evaluate_actions_distribution 返回的形状:", probs.shape)
                return probs
            except Exception as e:
                print("evaluate_actions_distribution 出错：", e)
        print("专家模型均不可用，使用 fallback one-hot 方法")
        actions, _ = expert_policy.predict(states.cpu().numpy(), deterministic=False)
        actions = torch.as_tensor(actions, device=states.device)
        batch_size = states.shape[0]
        num_actions = expert_policy.action_space.n
        probs = torch.zeros(batch_size, num_actions, device=states.device)
        for i, a in enumerate(actions):
            probs[i, a] = 1.0
        return probs.clamp(min=1e-7)

def get_agent_action_probs(agent, states):
    """
    获取 agent 在给定状态下的动作概率分布。
    优先尝试调用 agent.policy.get_distribution；如果没有，则尝试使用 evaluate_actions_distribution，
    如果仍然没有，则退回使用 agent.predict 构造 one-hot 分布。

    Args:
        agent: 当前 agent 模型（例如 PPO 模型）。
        states: Tensor，形状 [batch_size, state_dim]。

    Returns:
        Tensor，形状 [batch_size, num_actions]。
    """
    with torch.no_grad():
        if hasattr(agent.policy, "get_distribution"):
            try:
                dist = agent.policy.get_distribution(states)
                print("Agent 使用 get_distribution 获取概率分布")
                return dist.distribution.probs.clamp(min=1e-7)
            except Exception as e:
                print("Agent 的 get_distribution 出错：", e)
        if hasattr(agent.policy, "evaluate_actions"):
            try:
                print("Agent 使用 evaluate_actions_distribution 获取概率分布")
                probs = evaluate_actions_distribution(agent, states)
                print("Agent evaluate_actions_distribution 返回的形状:", probs.shape)
                return probs
            except Exception as e:
                print("Agent 的 evaluate_actions_distribution 出错：", e)
        print("Agent 均不可用，使用 fallback one-hot 方法")
        actions, _ = agent.predict(states.cpu().numpy(), deterministic=False)
        actions = torch.as_tensor(actions, device=states.device)
        batch_size = states.shape[0]
        num_actions = agent.action_space.n
        probs = torch.zeros(batch_size, num_actions, device=states.device)
        for i, a in enumerate(actions):
            probs[i, a] = 1.0
        return probs.clamp(min=1e-7)

def compute_kl_divergence(agent, expert_policy, states, device):
    """
    计算 KL 散度：KL(expert || agent) = Σ_a expert_prob(a|s)*log( expert_prob(a|s)/agent_prob(a|s) )

    Args:
        agent: 当前 agent 模型。
        expert_policy: 专家策略模型。
        states: NumPy 数组或列表，形状 [batch_size, state_dim]。
        device: 计算设备（例如 "cpu"）。

    Returns:
        平均 KL 散度（float）。
    """
    states = torch.FloatTensor(states).to(device)
    with torch.no_grad():
        expert_probs = get_expert_action_probs(expert_policy, states)
        agent_probs = get_agent_action_probs(agent, states)
        print("专家分布形状:", expert_probs.shape, "Agent 分布形状:", agent_probs.shape)
        kl_matrix = expert_probs * torch.log(expert_probs / agent_probs)
        kl_per_state = torch.sum(kl_matrix, dim=1)
        kl_mean = torch.mean(kl_per_state)
    return kl_mean.item()

def main():
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    
    model_zip_path = r"E:\TRRL\IQ-Learn\iq_learn\experts\cartpole.zip"
    if not os.path.exists(model_zip_path):
        print(f"模型文件 {model_zip_path} 不存在，请检查路径。")
        return
    
    # 加载专家策略
    expert_policy = load_ppo_from_zip(model_zip_path, env)
    if expert_policy is None:
        print("专家模型加载失败。")
        return
    
    # 加载 agent 模型（为了测试，这里同样加载 PPO，并添加扰动使其与专家不同）
    agent = load_ppo_from_zip(model_zip_path, env)
    if agent is None:
        print("Agent 模型加载失败。")
        return
    with torch.no_grad():
        for param in agent.policy.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    print("已对 agent 的策略参数添加扰动")
    
    # 采样一批状态；转换为 numpy 数组以提高效率
    states = [env.reset() for _ in range(128)]
    states = np.array(states)
    
    kl_value = compute_kl_divergence(agent, expert_policy, states, device)
    print(f"计算得到的 KL 散度： {kl_value:.6f}")

if __name__ == "__main__":
    main()
