import pickle
import pickle
import random

import argparse
import numpy as np

from pathlib import Path
import sys

def split_pickle_dict(input_path, output_prefix, seed=42):
    # Load the original pickle file
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Ensure it's a dictionary
    if not isinstance(data, dict):
        raise ValueError("Pickle file does not contain a dictionary.")

    # Shuffle keys
    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    # Split keys into 3 roughly equal parts
    n = len(keys)
    splits = [
        keys[:n // 3],
        keys[n // 3: 2 * n // 3],
        keys[2 * n // 3:]
    ]

    # Save each split
    for i, split_keys in enumerate(splits):
        split_dict = {k: data[k] for k in split_keys}
        file_path = f"{output_prefix}_part{i+1}.combinepkl"
        with open(file_path, 'wb') as f:
            pickle.dump(split_dict, f)
        print(f"Saved {len(split_dict)} items to {file_path}")

def split_pickle_dict_128000(input_path, output_prefix, seed=42):
    # Load the original pickle file
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Ensure it's a dictionary
    if not isinstance(data, dict):
        raise ValueError("Pickle file does not contain a dictionary.")

    # Determine key ranges
    ranges = [
        (0, 4000),
        (4001, 8000),
        (8001, 20000),
        (20001, 40000),
        (40001, 80000),
        (80001, 129000),

  
   
    ]

    # Prepare split dicts
    split_dicts = [{} for _ in range(len(ranges))]

    # Assign items to correct range
    for key, value in data.items():
        for i, (start, end) in enumerate(ranges):
            if start <= key <= end:
                split_dicts[i][key] = value
                break
            

    for i, d in enumerate(split_dicts):
        file_path = f"{output_prefix}_part{i+1}.combinepkl"
        with open(file_path, 'wb') as f:
            pickle.dump(d, f)
        print(f"Saved {len(d)} items to {file_path}")

    print("Splitting complete.")

import pickle
import torch

def combine_pickles(input_files, output_file, mode="tensor",filter=999999):
    combined = None

    if mode == "tensor":
        tensors = []
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not (isinstance(data, torch.Tensor) and data.ndim == 2):
                    raise ValueError(f"{file} does not contain a 2D torch tensor.")
                tensors.append(data)
                print(f"Loaded tensor {file} with shape {data.shape}")
        # Check shape[1] consistency
        shape1 = tensors[0].shape[1]
        if not all(t.shape[1] == shape1 for t in tensors):
            raise ValueError("All tensors must have the same shape[1].")
        combined = torch.cat(tensors, dim=0)
        print(f"Combined tensor shape: {combined.shape}")

    elif mode == "list":
        combined = []
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"{file} does not contain a list.")
                combined.extend(data)
                print(f"Loaded list {file} with {len(data)} elements")
        print(f"Combined list length: {len(combined)}")
    elif mode == "dict_replace":  # Assume no key conflicts
        combined = {}
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"{file} does not contain a dictionary.")
                for key, value in data.items():
                    combined[key] = value
                print(f"Loaded dict {file} with {len(data)} items")
        print(f"Combined dict length: {len(combined)}")

    elif mode == "dict_extend":  # Assume no key conflicts
        combined = {}
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"{file} does not contain a dictionary.")  
                for key, value in data.items():
                    if key not in combined:
                        combined[key] = []
                    combined[key].extend(value)
                print(f"Loaded dict {file} with {len(data)} items")
        print(f"Combined dict length: {len(combined)}")
    elif mode == "dict_nparray":
        combined = {}
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"{file} does not contain a dictionary.")
                for key, value in data.items():
                    if key not in combined:
                        combined[key] = value 
                    else:
                        combined[key] = np.vstack([combined[key], value])
        print(f"Combined dict length: {len(combined)}")
    elif mode == "dict_nparray_withfilter":
        combined = {}
        for file in input_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"{file} does not contain a dictionary.")
                for key, value in data.items():
                    if key not in combined:
                        combined[key] = value 
                    else:
                        combined[key] = np.vstack([combined[key], value])
        for k, v in combined.items():
            if v.shape[0] > filter:
                idx = np.random.choice(v.shape[0], filter, replace=False)  # pick unique rows
                combined[k] = v[idx]
    
        print(f"Combined dict length: {len(combined)}")
    
    else:
        raise ValueError("Mode not Found")

    # Save combined result
    with open(output_file, 'wb') as f:
        pickle.dump(combined, f)
    print(f"Saved combined output to {output_file}")


def collect_files(args):
    # Try folder first (if given)
    if args.input_folder:
        folder = Path(args.input_folder)
        if not folder.exists():
            sys.exit(f"Error: folder does not exist: {folder}")
        if not folder.is_dir():
            sys.exit(f"Error: not a directory: {folder}")

        folder_files = sorted(str(p) for p in folder.iterdir() if p.is_file())
        if folder_files:  # Non-empty => use these
            return folder_files

        # Folder is empty 
        if args.input_paths:
            return args.input_paths
        else:
            sys.exit("Error: folder is empty and no --input_paths provided.")

    #  No folder provided 
    if not args.input_paths:
        sys.exit("Error: provide either --input_folder or --input_paths.")
    return args.input_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Use nargs='+' to require at least one string
    parser.add_argument('-p','--input_paths', nargs='+', type=str, help='A list of strings')
    parser.add_argument('-f','--input_folder', type=str, help='A list of strings')
    parser.add_argument('-o','--output', type=str, default='./split_output')
    parser.add_argument('-m','--mode', type=str, default='tensor', help='Random seed for shuffling')
    parser.add_argument('--filter', type=int, default=999999, help='Filter for number of samples')


    args = parser.parse_args()

    input_path_list = collect_files(args)
    
    if args.mode == "split":
        split_pickle_dict(input_path_list[0], args.output)
    elif args.mode == "split128000":
        split_pickle_dict_128000(input_path_list[0], args.output)
    else:
        combine_pickles(input_path_list, args.output, mode=args.mode, filter=args.filter)

