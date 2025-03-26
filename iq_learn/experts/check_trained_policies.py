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

📂 Expected Folder Structure:
-----------------------------
Place all trained model files in a directory such as:
    E:/TRRL/IQ-Learn/iq_learn/trained_policies/

You can customize `folder_path` at the top of the script.

🛠 Example usage:
-----------------
Just run:
    python check_trained_policies.py

The script will recursively inspect all files in the specified directory.
"""


import os
import torch
import numpy as np
import pickle

folder_path = r"E:\TRRL\IQ-Learn\iq_learn\trained_policies"

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
            if data.dtype == 'object':
                print("Possibly contains a dict, trying to extract...")
                inner = data.item()
                print("Inner type:", type(inner))
                if isinstance(inner, dict):
                    print("Keys:", list(inner.keys()))
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

# Go through all files in the folder
for root, dirs, files in os.walk(folder_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        check_file(file_path)
