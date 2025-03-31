"""
Utility functions for computing KL divergence between policies, using
  KL(pi_E || pi_C) = E_{a ~ pi_E}[ log pi_E(a|s) - log pi_C(a|s) ]
in the discrete case, and a sampling-based approximation in the continuous case.
"""

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import os
import zipfile
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


def load_expert_policy_from_zip(model_zip_path, env, device="cpu", extract_dir="temp_model"):
    """
    Manually extract and load PPO model policy parameters from a zip file.
    
    Args:
        model_zip_path (str): Path to the zipped PPO model file
        env: Environment instance (to create a new PPO model)
        device: Device to load the model on
        extract_dir (str): Directory to extract files to
        
    Returns:
        Loaded PPO model or None if loading fails
    """
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    # Extract the zip file
    try:
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted files: {os.listdir(extract_dir)}")
    except Exception as e:
        print(f"Failed to extract zip file: {e}")
        return None
    
    # Look for policy.pth file
    policy_file = os.path.join(extract_dir, "policy.pth")
    if not os.path.exists(policy_file):
        print("policy.pth file not found in the extracted files")
        return None
    
    try:
        # Load policy parameters
        policy_state_dict = torch.load(policy_file, map_location=device)
        print(f"Loaded policy.pth with keys: {policy_state_dict.keys()}")
        
        # Create a new PPO model and load the parameters
        model = PPO("MlpPolicy", env, verbose=0)
        model.policy.to(device)
        model.policy.load_state_dict(policy_state_dict)
        print("Successfully loaded expert policy parameters")
        return model
    except Exception as e:
        print(f"Error loading policy parameters: {e}")
        return None


def compute_kl_divergence(agent, expert_policy, states, device):
    """
    Compute KL divergence between expert policy and agent policy for any environment
    using the formula: KL(pi_E || pi_C).
    
    - If environment is inferred to be discrete, we do:
         KL = sum_a pi_E(a|s) [log pi_E(a|s) - log pi_C(a|s)]
      and average over the batch of states.
    - If environment is inferred to be continuous, we approximate by sampling
      from the expert's Gaussian distribution for each state, then compute
         KL ~ mean( exp(log pi_E(a|s)) * [ log pi_E(a|s) - log pi_C(a|s) ] ).
    
    Args:
        agent: The IQ-Learn agent (has method get_q_values, or actor, etc.)
        expert_policy: Expert policy (SB3 model).
        states: States [batch_size, state_dim].
        device: Torch device.
        
    Returns:
        Mean KL divergence (float; returns abs() to avoid negative).
    """
    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # 简单的“有 action_dist 就认为是连续，否则离散”逻辑
        if hasattr(expert_policy, "policy") and hasattr(expert_policy.policy, "action_dist"):
            return abs(compute_continuous_kl_sampling(agent, expert_policy, states_t, device))
        else:
            return abs(compute_discrete_kl(agent, expert_policy, states_t, device))


def compute_discrete_kl(agent, expert_policy, states, device):
    """
    Discrete-action KL(pi_E || pi_C):
      = sum_a pi_E(a|s) [ log pi_E(a|s) - log pi_C(a|s) ].
      Returns abs(kl) at the end.
    """
    batch_size = states.shape[0]
    
    # 1) Get agent's action probabilities pi_C(a|s)
    if hasattr(agent, 'get_q_values'):
        q_vals = agent.get_q_values(states)
    elif hasattr(agent, 'getV'):
        q_vals = agent.getV(states, get_q=True)
    elif hasattr(agent, 'q_net'):
        q_vals = agent.q_net(states)
    else:
        raise RuntimeError("Agent doesn't have a method to get Q-values (for discrete).")
    
    agent_probs = F.softmax(q_vals, dim=-1).clamp(min=1e-7)
    action_dim = agent_probs.shape[1]
    
    # 2) Get expert's action probabilities pi_E(a|s)
    if (hasattr(expert_policy, "policy") and 
        hasattr(expert_policy.policy, "evaluate_actions")):
        expert_probs_list = []
        for action_idx in range(action_dim):
            # batch of that action
            actions_t = torch.full((batch_size, 1), action_idx, dtype=torch.long, device=device)
            try:
                _, logp_a, _ = expert_policy.policy.evaluate_actions(states, actions_t)
                p_a = torch.exp(logp_a)  # shape [batch_size, 1]
            except Exception as e:
                raise RuntimeError(f"Expert evaluate_actions failed for action {action_idx}: {e}")
            expert_probs_list.append(p_a)
        
        expert_probs = torch.cat(expert_probs_list, dim=1)  # shape [batch_size, action_dim]
    else:
        raise RuntimeError("Expert policy does not support 'evaluate_actions'; cannot compute discrete KL.")
    
    # 3) Convert to log-probs
    expert_probs = expert_probs.clamp(min=1e-7)
    agent_probs  = agent_probs.clamp(min=1e-7)
    lpE = torch.log(expert_probs)
    lpC = torch.log(agent_probs)

    # 4) KL(pi_E || pi_C) = sum_a pi_E(a|s)* [ log pi_E - log pi_C ]
    kl_matrix    = torch.exp(lpE) * (lpE - lpC)  # shape [batch_size, action_dim]
    kl_per_state = kl_matrix.sum(dim=1)          # sum over actions
    kl_mean      = kl_per_state.mean()           # average over batch

    return kl_mean.item()


def compute_continuous_kl_sampling(agent, expert_policy, states, device, num_samples=10):
    """
    Approximate KL(pi_E || pi_C) for continuous actions by sampling from 
    the expert distribution for each state. Returns a scalar float.
    """
    print("Computing continuous KL by sampling from expert distribution ...")
    batch_size = states.shape[0]

    with torch.no_grad():
        # 1) Expert distribution
        expert_mean, expert_log_std = get_mean_logstd(expert_policy, states)
        if expert_mean is None or expert_log_std is None:
            raise RuntimeError("Unable to get mean/log_std from expert PPO in continuous KL.")
        
        # 2) Agent distribution
        agent_mean, agent_log_std = None, None
        
        if hasattr(agent, 'actor'):
            # 可能是 SAC-style actor 或者别的
            actor_out = agent.actor(states)
            print(f"[DEBUG] agent.actor(...) returned => type: {type(actor_out)}")
            
            # 如果是 (mean, log_std) 直接解包
            if isinstance(actor_out, tuple) and len(actor_out) == 2:
                agent_mean, agent_log_std = actor_out
                print(f"[DEBUG] Actor returned tuple => mean={agent_mean.shape}, log_std={agent_log_std.shape}")
            else:
                # 否则它很可能是一个分布对象 (e.g. SquashedNormal)
                print("[DEBUG] Actor returned a distribution-like object:", actor_out)
                # 假定它有 base_dist.loc / scale
                if hasattr(actor_out, 'base_dist') and hasattr(actor_out.base_dist, 'loc'):
                    agent_mean    = actor_out.base_dist.loc
                    agent_log_std = actor_out.base_dist.scale.log()
                    print("[DEBUG] Extracted mean/log_std from base_dist:", 
                          agent_mean.shape, agent_log_std.shape)
                else:
                    raise RuntimeError("agent.actor returned an unknown object that we cannot parse as (mean, log_std).")

        elif (hasattr(agent, 'policy') and 
              hasattr(agent.policy, 'get_action_distribution')):
            # 另一种方式: SB3 PPO / etc
            distA = agent.policy.get_action_distribution(states)
            agent_mean    = distA.loc
            agent_log_std = distA.scale.log()
        else:
            raise RuntimeError("Agent does not have an actor or get_action_distribution for continuous KL.")
    
    # 3) shapes
    if agent_mean is None or agent_log_std is None:
        raise RuntimeError("Could not determine agent (mean, log_std) for continuous KL.")
    
    expert_std = torch.exp(expert_log_std)
    agent_std  = torch.exp(agent_log_std)
    action_dim = expert_mean.shape[1]

    # 4) Sample from Expert dist
    eps = torch.randn(num_samples, batch_size, action_dim, device=device)
    expanded_mean = expert_mean.unsqueeze(0)
    expanded_std  = expert_std.unsqueeze(0)
    sampled_actions = expanded_mean + expanded_std * eps  # [num_samples, batch_size, action_dim]

    # 5) compute log pi_E(a) & log pi_C(a)
    logpE = compute_diag_gaussian_log_prob(sampled_actions, expert_mean, expert_std)
    logpC = compute_diag_gaussian_log_prob(sampled_actions, agent_mean,  agent_std)

    # 6) KL ~ mean( exp(log pE) * [ log pE - log pC ] )
    pointwise_kl = torch.exp(logpE) * (logpE - logpC)  # shape [num_samples, batch_size]
    kl_val = pointwise_kl.mean().item()
    
    return kl_val


def get_mean_logstd(sb3_model, states):
    """
    Attempt to extract (mean, log_std) from an older SB3 PPO model's policy,
    by manually using its internal _get_latent + action_net + log_std logic.
    Raises RuntimeError if something goes wrong.
    """
    if not hasattr(sb3_model, 'policy'):
        raise RuntimeError("SB3 model does not have a policy attribute.")
    policy = sb3_model.policy

    # 如果 policy 有 get_distribution，可以用更现代的方式
    # 但假设当前是“老版本” => 继续用 _get_latent + action_net
    if (not hasattr(policy, '_get_latent') or
        not hasattr(policy, 'action_net') or
        not hasattr(policy, 'log_std')):
        raise RuntimeError("Policy does not expose the needed attributes for mean/log_std extraction.")

    with torch.no_grad():
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        try:
            # 1) 获取 policy 的 latent
            latent_pi, _, _ = policy._get_latent(states)
            # 2) 计算 mean
            mean = policy.action_net(latent_pi)  # shape [batch_size, action_dim]
            # 3) 从 policy.log_std 里取 log_std 并 broadcast
            log_std = policy.log_std  # shape [action_dim]
            while log_std.dim() < mean.dim():
                log_std = log_std.unsqueeze(0)
            log_std = log_std.expand_as(mean)

            return mean, log_std
        except Exception as e:
            raise RuntimeError(f"Unable to extract (mean, log_std) from older PPO policy: {e}")


def compute_diag_gaussian_log_prob(actions, mean, std):
    """
    Compute log prob under diagonal Gaussian N(mean, std^2),
    for 'actions' shape: [num_samples, batch_size, action_dim].
    'mean', 'std' shape: [batch_size, action_dim].
    Return shape: [num_samples, batch_size].
    """
    num_samples, batch_size, action_dim = actions.shape
    expanded_mean = mean.unsqueeze(0).expand(num_samples, batch_size, action_dim)
    expanded_std  = std.unsqueeze(0).expand(num_samples, batch_size, action_dim)
    
    # log N(a|m,s) = -sum_i [ 0.5*((a-m)/s)^2 + log(s * sqrt(2*pi)) ]
    var = expanded_std ** 2
    log_probs = -0.5 * ((actions - expanded_mean)**2 / var).sum(dim=2)
    log_probs = log_probs - (expanded_std.log() + 0.5 * np.log(2*np.pi)).sum(dim=2)
    return log_probs


def load_expert_policy(expert_path, env_name, device="cpu", env=None):
    """
    Load expert policy using the manual extraction approach for any environment.
    No mock fallback; if not found => return None.
    """
    env_name_normalized = env_name.lower()

    if os.path.isdir(expert_path):
        possible_paths = [
            os.path.join(expert_path, f"{env_name_normalized}.zip"),
            os.path.join(expert_path, f"{env_name}.zip"),
            os.path.join(expert_path, f"expert_{env_name_normalized}.zip"),
            os.path.join(expert_path, f"expert_{env_name}.zip")
        ]
        for path in possible_paths:
            if os.path.exists(path):
                expert_path = path
                break
    
    if os.path.exists(expert_path):
        try:
            print(f"Attempting to load expert from {expert_path}")
            return load_expert_policy_from_zip(expert_path, env, device)
        except Exception as e:
            print(f"Failed to load expert from {expert_path}: {e}")
    
    # If none worked, just return None (no mock).
    print("No expert model found.")
    return None
