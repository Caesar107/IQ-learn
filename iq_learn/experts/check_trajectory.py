"""
check_trajectory.py

This script is used to **analyze expert trajectory data** stored in a `.npy` file.
Specifically, it loads `transitions_CartPole-v1.npy`, which is expected to contain
a list of expert demonstration trajectories (episodes), where each trajectory is
a sequence of transitions (e.g., (state, action, reward, next_state, done)).

The script will:
- Count how many expert episodes are stored
- Compute the length (i.e., number of steps) of each episode
- Print summary statistics: mean, standard deviation, max, and min length

Typical use case:
- Verify the quality and consistency of your expert dataset before training
- Understand whether episodes are long enough or too short
- Debug data preprocessing issues
"""
import numpy as np

def main():
    path = 'iq_learn/experts/Acrobot_expert_trajs.npy'
    print("Loading file:", path)
    data = np.load(path, allow_pickle=True)

    print("Loaded type:", type(data))
    print("Shape:", getattr(data, "shape", "No shape"))

    if isinstance(data, np.ndarray) and data.shape == ():
        print("Scalar ndarray detected, extracting...")
        data = data.item()
        print("After item(), type:", type(data))

    try:
        print(f"Loaded {len(data)} trajectories from {path}")
    except TypeError:
        print("Cannot take len() of this object.")
    lengths = [len(traj) for traj in data["states"]]

    print("Trajectory lengths:")
    for i, l in enumerate(lengths):
        print(f"  Trajectory {i+1}: {l} steps")

    print("\nStats:")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Std : {np.std(lengths):.2f}")
    print(f"  Max : {np.max(lengths)}")
    print(f"  Min : {np.min(lengths)}")


if __name__ == "__main__":
    main()

