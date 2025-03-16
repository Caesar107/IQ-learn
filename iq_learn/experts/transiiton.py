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
