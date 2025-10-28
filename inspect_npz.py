#!/usr/bin/env python3
import argparse
import numpy as np
import pprint

def main():
    parser = argparse.ArgumentParser(description="Pretty-print the contents of a .npz file")
    parser.add_argument("path", help="Path to the .npz file")
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)
    print(f"Loaded keys: {list(data.keys())}\n")

    for key in data.files:
        arr = data[key]
        print(f"--- {key} ---")
        print(f"shape: {arr.shape}, dtype: {arr.dtype}")
        pprint.pprint(arr.tolist() if arr.size < 50 else arr[:10].tolist())
        print()

if __name__ == "__main__":
    main()