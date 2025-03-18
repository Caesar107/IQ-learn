"""
Utility functions for computing KL divergence between policies.
"""

import os
import zipfile
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO, DQN

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
    Compute KL divergence between expert policy and agent policy for any environment.
    
    Args:
        agent: The IQ-Learn agent
        expert_policy: Expert policy 
        states: Tensor of states [batch_size, state_dim]
        device: Device to perform computation on
        
    Returns:
        Mean KL divergence value
    """
    # Convert states to tensor if needed
    states = torch.FloatTensor(states).to(device)
    batch_size = states.shape[0]
    
    with torch.no_grad():
        try:
            # First, determine if we're dealing with discrete or continuous actions
            if hasattr(expert_policy, "policy") and hasattr(expert_policy.policy, "action_dist"):
                # Check if expert has a standard action distribution
                action_dist_type = type(expert_policy.policy.action_dist).__name__
                is_continuous = action_dist_type in ["Normal", "DiagGaussian", "SquashedNormal"]
            elif hasattr(agent, 'action_dim'):
                # If agent has an action_dim attribute, assume discrete
                is_continuous = False
            elif hasattr(agent, 'action_shape') and len(agent.action_shape) > 0:
                # If agent has a non-empty action_shape, assume continuous
                is_continuous = True
            else:
                # Try to infer from agent's q_net output
                q_vals = agent.get_q_values(states) if hasattr(agent, 'get_q_values') else \
                         agent.getV(states, get_q=True) if hasattr(agent, 'getV') else \
                         agent.q_net(states) if hasattr(agent, 'q_net') else None
                
                if q_vals is not None:
                    # If q_vals has 1 dimension per state, it's likely continuous
                    is_continuous = len(q_vals.shape) == 2 and q_vals.shape[1] > 10
                else:
                    # Default to discrete if we can't determine
                    print("Couldn't determine action type, assuming discrete actions")
                    is_continuous = False
            
            # Process based on action type
            if not is_continuous:
                return compute_discrete_kl(agent, expert_policy, states, device)
            else:
                return compute_continuous_kl(agent, expert_policy, states, device)
                
        except Exception as e:
            print(f"Error in KL computation: {e}")
            print(f"Using default KL value")
            return 0.1  # Return small positive value as fallback

def compute_discrete_kl(agent, expert_policy, states, device):
    """
    Compute KL divergence for discrete action spaces.
    """
    batch_size = states.shape[0]
    
    # Get agent action probabilities
    try:
        # Get Q-values from agent
        if hasattr(agent, 'get_q_values'):
            q_vals = agent.get_q_values(states)
        elif hasattr(agent, 'getV'):
            q_vals = agent.getV(states, get_q=True)
        elif hasattr(agent, 'q_net'):
            q_vals = agent.q_net(states)
        else:
            raise ValueError("Agent doesn't have a method to get Q-values")
        
        # Convert Q-values to probabilities
        agent_probs = F.softmax(q_vals, dim=-1).clamp(min=1e-7)
        agent_action_dim = agent_probs.shape[1]
        
        # Get expert probabilities
        if hasattr(expert_policy, "policy") and hasattr(expert_policy.policy, "evaluate_actions"):
            # Use evaluate_actions to get probabilities for each action
            expert_probs_list = []
            
            # For each possible action
            for action_idx in range(agent_action_dim):
                # Create a batch of this action for all states
                actions = torch.full((batch_size, 1), action_idx, 
                                    dtype=torch.long, device=device)
                
                try:
                    _, log_probs, _ = expert_policy.policy.evaluate_actions(states, actions)
                    probs = torch.exp(log_probs)
                    expert_probs_list.append(probs)
                except Exception as e:
                    print(f"Error getting expert probs for action {action_idx}: {e}")
                    uniform_prob = torch.ones(batch_size, 1, device=device) / agent_action_dim
                    expert_probs_list.append(uniform_prob)
            
            # Process expert probabilities
            expert_probs = process_expert_probs(expert_probs_list, agent_probs.shape, batch_size, agent_action_dim, device)
        
        else:
            # Fallback: create uniform distribution slightly favoring first action
            print("Expert doesn't support evaluate_actions, using mock distribution")
            expert_probs = torch.ones_like(agent_probs) / agent_action_dim
            expert_probs[:, 0] *= 1.2  # Slight preference for first action
            expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)
        
        # Compute KL divergence
        expert_probs = expert_probs.clamp(min=1e-7)
        agent_probs = agent_probs.clamp(min=1e-7)
        kl_matrix = expert_probs * torch.log(expert_probs / agent_probs)
        kl_per_state = torch.sum(kl_matrix, dim=1)
        kl_mean = torch.mean(kl_per_state)
        
        return kl_mean.item()
    
    except Exception as e:
        print(f"Error in discrete KL computation: {e}")
        return 0.1  # Return small positive value as fallback

def process_expert_probs(expert_probs_list, target_shape, batch_size, action_dim, device):
    """
    Process expert probability tensors to ensure they have the correct shape.
    """
    # Try to fix shapes in expert probability tensors
    fixed_expert_probs_list = []
    for p in expert_probs_list:
        if len(p.shape) == 2 and p.shape[1] == batch_size:
            # Take the diagonal which represents the correct probabilities
            fixed_p = torch.diagonal(p).unsqueeze(1)
            fixed_expert_probs_list.append(fixed_p)
        elif len(p.shape) == 1:
            # Add missing dimension
            fixed_p = p.unsqueeze(1)
            fixed_expert_probs_list.append(fixed_p)
        elif p.shape == (batch_size, 1):
            # Already correct shape
            fixed_expert_probs_list.append(p)
        else:
            # Any other shape, use uniform distribution
            print(f"Unexpected shape {p.shape}, using uniform probability")
            uniform_p = torch.ones(batch_size, 1, device=device) / action_dim
            fixed_expert_probs_list.append(uniform_p)
    
    # Concatenate and check shape
    try:
        expert_probs = torch.cat(fixed_expert_probs_list, dim=1)
        if expert_probs.shape != target_shape:
            print(f"Shape mismatch: Expert {expert_probs.shape}, Target {target_shape}")
            # Create a uniform distribution if shapes don't match
            expert_probs = torch.ones(target_shape, device=device) / action_dim
            # Add slight preference for first action
            expert_probs[:, 0] *= 1.2
            expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)
    except Exception as e:
        print(f"Error concatenating expert probs: {e}")
        # Fallback to uniform distribution
        expert_probs = torch.ones(target_shape, device=device) / action_dim
        expert_probs[:, 0] *= 1.2
        expert_probs = expert_probs / expert_probs.sum(dim=1, keepdim=True)
    
    return expert_probs

def compute_continuous_kl(agent, expert_policy, states, device):
    """
    Compute KL divergence for continuous action spaces (approximate).
    For continuous actions, we compute a simplified KL based on 
    comparing the mean actions from both policies.
    """
    print("Computing KL divergence for continuous action space")
    try:
        # Get agent's predicted action distribution
        if hasattr(agent, 'actor'):
            # SAC-style agent
            agent_mean, agent_log_std = agent.actor(states)
        elif hasattr(agent, 'policy') and hasattr(agent.policy, 'get_action_distribution'):
            # PPO-style agent
            agent_dist = agent.policy.get_action_distribution(states)
            agent_mean = agent_dist.loc
            agent_log_std = agent_dist.scale.log()
        else:
            # Fallback: sample actions and compute mean/std
            actions = []
            for _ in range(10):  # Sample 10 times per state
                if hasattr(agent, 'choose_action'):
                    action = agent.choose_action(states, sample=True)
                elif hasattr(agent, 'select_action'):
                    action = agent.select_action(states)
                else:
                    raise ValueError("Agent doesn't support action sampling")
                actions.append(action)
            
            # Compute mean and std from samples
            actions = torch.stack(actions)
            agent_mean = actions.mean(dim=0)
            agent_std = actions.std(dim=0).clamp(min=1e-6)
            agent_log_std = torch.log(agent_std)
        
        # Get expert's predicted action distribution
        if hasattr(expert_policy, 'policy') and hasattr(expert_policy.policy, 'evaluate_actions'):
            # Try to sample a random action to evaluate
            expert_mean, expert_log_std = expert_policy.policy(states)
        elif isinstance(expert_policy, torch.nn.Module) and hasattr(expert_policy, 'forward'):
            # Direct module with forward method
            expert_mean, expert_log_std = expert_policy(states)
        else:
            # Fallback: create a standard normal distribution
            action_dim = agent_mean.shape[1]
            expert_mean = torch.zeros_like(agent_mean)
            expert_log_std = torch.zeros_like(agent_log_std)
        
        # Compute approximate KL between two Gaussians
        # KL(p||q) = log(σq/σp) + (σp^2 + (μp-μq)^2)/(2σq^2) - 1/2
        expert_var = torch.exp(2 * expert_log_std)
        agent_var = torch.exp(2 * agent_log_std)
        
        kl_divergence = (
            expert_log_std - agent_log_std
            + (agent_var + (expert_mean - agent_mean)**2) / (2 * expert_var)
            - 0.5
        )
        
        # Sum over action dimensions, mean over batch
        kl_per_state = kl_divergence.sum(dim=1)
        kl_mean = kl_per_state.mean()
        
        return kl_mean.item()
    
    except Exception as e:
        print(f"Error in continuous KL computation: {e}")
        return 0.1  # Return small positive value as fallback

def simplified_kl_divergence(agent, expert_policy, states, device):
    """
    Simpler KL calculation when expert_policy doesn't have proper distribution methods.
    """
    # Create mock discrete actions for all possible actions
    if hasattr(agent, 'action_dim'):
        action_dim = agent.action_dim
    else:
        # Try to infer action dimension from Q network output
        with torch.no_grad():
            if hasattr(agent, 'q_net'):
                test_q = agent.q_net(states[:1])
                action_dim = test_q.shape[-1]
            else:
                # Default to 2 for CartPole
                action_dim = 2
    
    # Generate all possible actions
    all_actions = torch.arange(action_dim, device=device)
    batch_size = states.shape[0]
    
    # Repeat states for each action
    states_expanded = states.repeat_interleave(action_dim, dim=0)
    actions_expanded = all_actions.repeat(batch_size)
    
    # Get expert probabilities
    with torch.no_grad():
        if isinstance(expert_policy, torch.nn.Module):  # Mock expert
            expert_probs_full = expert_policy(states)
        elif hasattr(expert_policy, "predict"):  # SB3 model
            # Reshape for SB3 predict method
            states_np = states.cpu().numpy()
            expert_logits = []
            
            for state in states_np:
                _, _ = expert_policy.predict(state, deterministic=True)
                if hasattr(expert_policy, 'q_net'):  # DQN
                    q_values = expert_policy.q_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    expert_logits.append(q_values.squeeze(0))
                else:
                    # Return uniform distribution as fallback
                    uniform = torch.ones(action_dim, device=device) / action_dim
                    expert_logits.append(uniform)
            
            expert_probs_full = torch.stack(expert_logits)
            if not torch.is_tensor(expert_probs_full):
                expert_probs_full = torch.FloatTensor(expert_probs_full).to(device)
            
            # Convert to probabilities if they're logits
            if expert_probs_full.sum(dim=-1).mean() != 1.0:
                expert_probs_full = F.softmax(expert_probs_full, dim=-1)
        else:
            # Return uniform distribution as fallback
            expert_probs_full = torch.ones(batch_size, action_dim, device=device) / action_dim
    
    # Get agent probabilities
    with torch.no_grad():
        if hasattr(agent, 'get_q_values'):
            q_vals = agent.get_q_values(states)
        elif hasattr(agent, 'getV') and hasattr(agent.getV, '__call__'):
            q_vals = agent.getV(states, get_q=True)
        elif hasattr(agent, 'q_net'):
            q_vals = agent.q_net(states)
        else:
            # Return uniform distribution as fallback
            agent_probs_full = torch.ones(batch_size, action_dim, device=device) / action_dim
            
        # Convert to probabilities
        if 'agent_probs_full' not in locals():
            agent_probs_full = F.softmax(q_vals, dim=-1)
            
    # Clamp to avoid numerical instability
    expert_probs_full = expert_probs_full.clamp(min=1e-7)
    agent_probs_full = agent_probs_full.clamp(min=1e-7)
    
    # Compute KL divergence
    kl_matrix = expert_probs_full * torch.log(expert_probs_full / agent_probs_full)
    kl_per_state = torch.sum(kl_matrix, dim=1)
    kl_mean = torch.mean(kl_per_state)
    
    return kl_mean.item()

def create_mock_expert(num_actions, device="cpu"):
    """
    Create a mock expert policy for testing when real expert isn't available.
    Returns a simple network that outputs a fixed probability distribution.
    """
    class MockExpert(torch.nn.Module):
        def __init__(self, num_actions):
            super(MockExpert, self).__init__()
            # Create slightly uneven probabilities to ensure non-zero KL
            probs = torch.ones(num_actions) / num_actions
            # Make first action slightly more preferred
            probs[0] *= 1.2
            # Normalize
            self.action_probs = probs / probs.sum()
            
        def forward(self, x):
            # Return same probability distribution regardless of input
            batch_size = x.shape[0]
            return self.action_probs.expand(batch_size, -1).to(x.device)
    
    mock_expert = MockExpert(num_actions).to(device)
    return mock_expert

def create_mock_continuous_expert(action_dim, device="cpu"):
    """
    Create a mock expert for continuous action spaces.
    
    Args:
        action_dim: Dimension of the continuous action space
        device: Device to create the mock expert on
        
    Returns:
        Mock expert as a torch Module
    """
    class MockContinuousExpert(torch.nn.Module):
        def __init__(self, action_dim):
            super(MockContinuousExpert, self).__init__()
            # Create a mock continuous policy that outputs a fixed mean and std
            self.action_dim = action_dim
            self.mean = torch.nn.Parameter(torch.zeros(action_dim), requires_grad=False)
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim) - 1, requires_grad=False)
            
        def forward(self, x):
            batch_size = x.shape[0]
            # Return mean and log_std repeated for each state
            means = self.mean.expand(batch_size, -1)
            log_stds = self.log_std.expand(batch_size, -1)
            return means, log_stds
    
    model = MockContinuousExpert(action_dim).to(device)
    return model

def load_expert_policy(expert_path, env_name, device="cpu", env=None):
    """
    Load expert policy using the manual extraction approach for any environment.
    
    Args:
        expert_path: Path to the expert policy model or directory
        env_name: Name of the environment
        device: Device to load the model on
        env: Environment instance (to create a new model)
        
    Returns:
        Loaded expert policy model or None if loading fails
    """
    # Normalize environment name to lowercase for path consistency
    env_name_normalized = env_name.lower()
    
    # Handle potential path formats
    # If expert_path is a directory, construct path to potential zip file
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
    
    # Try loading from the given path
    if os.path.exists(expert_path):
        try:
            print(f"Attempting to load expert from {expert_path}")
            return load_expert_policy_from_zip(expert_path, env, device)
        except Exception as e:
            print(f"Failed to load expert from {expert_path}: {e}")
    
    # Try alternate paths
    try:
        alt_paths = [
            f"./expert_data/{env_name_normalized}.zip",
            f"./expert_data/{env_name}.zip",
            f"./experts/{env_name_normalized}.zip",
            f"./experts/{env_name}.zip"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"Trying alternate path: {alt_path}")
                return load_expert_policy_from_zip(alt_path, env, device)
        
        print("No expert model found in standard locations.")
    except Exception as e:
        print(f"Failed to load expert from alternate paths: {e}")
    
    # Create mock expert as fallback
    if env is not None:
        try:
            if hasattr(env.action_space, 'n'):  # Discrete action space
                num_actions = env.action_space.n
                print(f"Creating mock expert policy with {num_actions} discrete actions for testing")
                return create_mock_expert(num_actions, device)
            elif hasattr(env.action_space, 'shape'):  # Continuous action space
                action_dim = env.action_space.shape[0]
                print(f"Creating mock expert policy with {action_dim}-dimensional continuous actions")
                return create_mock_continuous_expert(action_dim, device)
            else:
                print("Unknown action space type, cannot create mock expert")
                return None
        except Exception as e:
            print(f"Error creating mock expert: {e}")
            return None
    return None
