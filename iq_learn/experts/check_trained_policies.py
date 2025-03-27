"""
check_trained_policies.py

This script is used to **inspect and validate saved policy/model files** in a given folder.
It attempts to load each file using multiple common serialization methods and prints out
basic information such as file size, data type, and top-level keys (if available).

🔍 What it does:
---------------
For each file under the specified `folder_path`, it tries to:
1. Print the file size (in MB)
2. Load the file with `torch.load()` — typically for PyTorch model checkpoints
3. Load the file with `np.load()` — typically for `.npy` or `.npz` arrays
4. Load the file with `pickle.load()` — fallback for generic Python object files
5. Report the type and structure of the loaded object (e.g., dict keys)

✅ Use cases:
------------
- To check what's inside a saved model or policy file
- To verify whether a file is loadable and by which method
- To debug loading errors or understand legacy formats
- To prepare for converting or reusing trained policies

📂 Expected Input:
-----------------------------
Can be either:
- A directory path containing model files
- A direct path to a single model file

🛠 Example usage:
-----------------
Just run:
    python check_trained_policies.py
"""


import os
import torch
import numpy as np
import pickle
import sys

# Path to check - can be a directory or a single file
path_to_check = r"/home/yche767/IQ-learn/iq_learn/experts/Ant-v4_expert_trajs.npy"

def check_file(file_path):
    print(f"\nChecking file: {file_path}")

    # First, print file size
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")

    # Try loading with torch
    try:
        data = torch.load(file_path, map_location='cpu')
        print("Loaded with torch ✅")
        if isinstance(data, dict):
            print("Type: dict")
            print("Keys:", list(data.keys()))
        else:
            print("Type:", type(data))
    except Exception as e:
        print(f"torch.load failed ❌: {e}")

    # Try loading with numpy
    try:
        data = np.load(file_path, allow_pickle=True)
        print("Loaded with numpy ✅")
        if isinstance(data, np.ndarray):
            print("Type: numpy.ndarray")
            print("Shape:", data.shape)
            print("Data type:", data.dtype)
            if data.dtype == 'object':
                print("Possibly contains a dict, trying to extract...")
                try:
                    inner = data.item()
                    print("Inner type:", type(inner))
                    if isinstance(inner, dict):
                        print("Keys:", list(inner.keys()))
                except:
                    print("Could not extract item from array")
            else:
                # Print a sample of the array if it's not too big
                if data.size < 100:
                    print("Sample data:", data[:10])
                else:
                    print("Sample data:", data.flatten()[:10], "...")
        else:
            print("Type:", type(data))
    except Exception as e:
        print(f"np.load failed ❌: {e}")

    # Try loading with pickle
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Loaded with pickle ✅")
        print("Type:", type(data))
        if isinstance(data, dict):
            print("Keys:", list(data.keys()))
    except Exception as e:
        print(f"pickle.load failed ❌: {e}")

def main():
    if os.path.isfile(path_to_check):
        # If path is a single file
        check_file(path_to_check)
    elif os.path.isdir(path_to_check):
        # If path is a directory
        for root, dirs, files in os.walk(path_to_check):
            for filename in files:
                file_path = os.path.join(root, filename)
                check_file(file_path)
    else:
        print(f"Error: Path '{path_to_check}' does not exist!")
        return

if __name__ == "__main__":
    main()
