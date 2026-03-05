import pickle
import torch
from tqdm import tqdm
import argparse
import os
import numpy as np
import math

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_names', type=str, nargs='+', default=['./CLS_encoding_traint/'],
                        help="List of folder names to search for pkl files")
    parser.add_argument('--output_path', type=str, default='./CLS_encoding_traint/CLS.combinepkl',
                        help="Output pkl file keyword (base path/prefix, no extension)")
    parser.add_argument('--num_splits', type=int, default=3,
                        help="How many even splits to produce after combining")
    args = parser.parse_args()

    cache_dict = {}

    # Combine all pkl files into cache_dict
    for folder in args.folder_names:
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if filename.endswith('.pkl'):
                    file_path = os.path.join(root, filename)
                    with open(file_path, 'rb') as file:
                        data = pickle.load(file)
                        print(f"Loaded data from {file_path}")

                    for key, value in data.items():
                        cache_dict.setdefault(key, [])
                        cache_dict[key].extend(value)

    # Stack each list into a single numpy array
    for key, value in cache_dict.items():
        if not (isinstance(value, list) and all(isinstance(x, np.ndarray) for x in value)):
            value = [np.array(x) for x in value]
        cache_dict[key] = np.vstack(value) if len(value) > 0 else np.empty((0,))

    # Split after combining: each output file has all keys, with ~1/n rows per key
    n = args.num_splits
    base = args.output_path  

    # Precompute splits for each key to avoid re-splitting in the loop
    split_buckets_per_key = {}
    #  Assign keys to splits instead of splitting rows
    key_sizes = {key: cache_dict[key].shape[0] for key in cache_dict}
    sorted_keys = sorted(key_sizes.items(), key=lambda x: x[1], reverse=True)

    buckets = [dict() for _ in range(n)]
    bucket_sizes = [0] * n

    for key, size in sorted_keys:
        # assign to the bucket with the smallest current load
        idx = int(np.argmin(bucket_sizes))
        buckets[idx][key] = cache_dict[key]
        bucket_sizes[idx] += size

    #  Save 
    for i, part_dict in enumerate(buckets):
        out_path = f"{base}.part{i+1}.combinepkl"
        print(f"Saving split {i+1}/{n} ({len(part_dict)} keys, {bucket_sizes[i]} rows) to {out_path}")
        with open(out_path, 'wb') as handle:
            pickle.dump(part_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
