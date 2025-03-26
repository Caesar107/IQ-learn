import torch
import numpy as np
import argparse

# Parse command-line arguments
#python convert_transitions.py --env_name acrobot
#python convert_transitions.py --env_name LunarLander-v2 BipedalWalker-v3
parser = argparse.ArgumentParser(description='Convert transitions to expert trajectories.')
parser.add_argument('--env_name', type=str, required=True, help='Name of the environment')
args = parser.parse_args()

# Load transitions from file
transitions_file = f'./experts/transitions_{args.env_name}.npy'
transitions = torch.load(transitions_file)

# Print the structure of the transitions
print("Transitions structure:", type(transitions))
if hasattr(transitions, 'obs'):
    print("Number of observations:", len(transitions.obs))
    print("Number of actions:", len(transitions.acts))
    print("Number of next observations:", len(transitions.next_obs))
    print("Number of dones:", len(transitions.dones))

# Initialize expert episodes
episodes = []
episode_states = []
episode_actions = []
episode_next_states = []
episode_rewards = []
episode_dones = []

# Process transitions
for i in range(len(transitions.obs)):
    state = transitions.obs[i]
    next_state = transitions.next_obs[i]
    action = transitions.acts[i]
    reward = transitions.rewards[i] if hasattr(transitions, 'rewards') else 0  # Default to 0 if not present
    done = transitions.dones[i]
    
    episode_states.append(state)
    episode_actions.append(action)
    episode_next_states.append(next_state)
    episode_rewards.append(reward)
    episode_dones.append(done)
    
    if done:
        episode = {
            "states": np.array(episode_states),
            "actions": np.array(episode_actions),
            "next_states": np.array(episode_next_states),
            "rewards": np.array(episode_rewards),
            "dones": np.array(episode_dones)
        }
        
        episodes.append(episode)
        
        episode_states = []
        episode_actions = []
        episode_next_states = []
        episode_rewards = []
        episode_dones = []

# Initialize expert trajectories dictionary
expert_trajs = {
    "states": [],
    "actions": [],
    "next_states": [],
    "rewards": [],
    "dones": [],
    "lengths": []
}

# Fill expert trajectories dictionary with episode data
for episode in episodes:
    expert_trajs["states"].append(episode["states"])
    expert_trajs["actions"].append(episode["actions"])
    expert_trajs["next_states"].append(episode["next_states"])
    expert_trajs["rewards"].append(episode["rewards"])
    expert_trajs["dones"].append(episode["dones"])
    expert_trajs["lengths"].append(len(episode["states"]))

# Print lengths of the episodes
print("Lengths of the episodes:", expert_trajs["lengths"])

# Save expert trajectories as NumPy array
output_file = f'./experts/{args.env_name}_expert_trajs.npy'
np.save(output_file, expert_trajs, allow_pickle=True)

print('Expert trajectories saved successfully in ExpertDataset-compatible format.')
