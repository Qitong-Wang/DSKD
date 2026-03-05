import pickle
import torch
from tqdm import tqdm
import argparse
import os
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Allow multiple folder names
    parser.add_argument('--add_file', type=str,
                        help="List of folder names to search for pkl files")
    parser.add_argument('--output_path', type=str, default='./CLS_encoding_traint/CLS.combinepkl',
                        help="Output pkl file keyword")
    parser.add_argument('--record_txt', type=str, nargs='+', required=True,
                        help="List of txt files containing indices to remove")
    args = parser.parse_args()

    # Load the pickle file
    with open(args.add_file, 'rb') as file:
        add_data = pickle.load(file) 
        print(f"Loaded data from {args.add_file}")
    
    # Collect all indices to remove from the txt files
    indices_to_remove = set()
    for txt_file in args.record_txt:
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                for line in f:
                    # Each line might contain one or more indices
                    line = line.strip()
                    if line:
                        # Split by whitespace and convert to integers
                        try:
                            line_indices = [int(idx) for idx in line.split()]
                            indices_to_remove.update(line_indices)
                        except ValueError:
                            print(f"Warning: Could not parse line '{line}' in {txt_file}")
            print(f"Loaded indices from {txt_file}")
        else:
            print(f"Warning: File {txt_file} does not exist")
    
    print(f"Total indices to remove: {len(indices_to_remove)}")
    
    # Filter the data based on indices
    filtered_data = {}
    
   
    original_count = len(add_data)
    
    # Remove entries with indices that are in indices_to_remove
    for key in add_data.keys():
        if key not in indices_to_remove:
            filtered_data[key] = add_data[key]
            print(f"Keeping key: {key}, shape: {add_data[key].shape}")
    
    print(f"Original data count: {original_count}")
    print(f"Filtered data count: {len(filtered_data)}")
    print(f"Removed {original_count - len(filtered_data)} entries")
    
    # Save only if there are remaining entries
    if filtered_data:
        output_path = args.output_path
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print("Saving filtered data to", output_path)
        with open(output_path, 'wb') as handle:
            pickle.dump(filtered_data, handle)
        print("Filtering and saving completed successfully!")
    else:
        print("No data remaining after filtering. Output file not created.")
