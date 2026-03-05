import pickle
import torch
from tqdm import tqdm
import argparse
import os
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Allow multiple folder names
    parser.add_argument('--base_file', type=str,
                        help="List of folder names to search for pkl files")
    parser.add_argument('--add_file', type=str,
                        help="List of folder names to search for pkl files")
    parser.add_argument('--add_file_num', type=int, default=2500,
                        help="Number of records to add from the add_file")
    parser.add_argument('--output_path', type=str,
                        help="Output pkl file keyword")
    parser.add_argument('--add_record_txt', type=str, default='./output_txt',
                        help="Output pkl file keyword")
    args = parser.parse_args()

    # Open the .pkl file
    with open(args.base_file, 'rb') as file:
        data = pickle.load(file) 
        print(f"Loaded data from {args.base_file}")

    with open(args.add_file, 'rb') as file:
        add_data = pickle.load(file) 
        print(f"Loaded data from {args.add_file}")

    record_list = []
    # Process the data and create tensors

    for key, value in add_data.items():
        if key in data:
            if args.add_file_num < value.shape[0]:
                idx = np.random.choice(value.shape[0], args.add_file_num, replace=False)
                selected = value[idx]
            else:
                selected = value

            if args.add_file_num < data[key].shape[0]:
                idx2 = np.random.choice(data[key].shape[0], args.add_file_num, replace=False)
                data[key] = data[key][idx2]

            print(f"Combined data for key: {key}, base shape: {data[key].shape}, add shape: {selected.shape}")
            data[key] = np.vstack((data[key], selected))
            record_list.append(key)
    with open(args.add_record_txt, 'w') as f:
        for item in record_list:
            f.write("%s\n" % item)

     
    # Save the combined data to an output file
    output_path = args.output_path
    print("Saving combined data to", output_path)
    with open(output_path, 'wb') as handle:
        pickle.dump(data, handle)
