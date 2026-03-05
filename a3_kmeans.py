
import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./STS_encoding_test/combine_1000_index2994.combinepkl',
                        help="Path to the input .combinepkl file containing at most 1000 encodings per token.")
    parser.add_argument('--output_path', type=str, default='./llama3_STS16_fp32freq',
                        help="Path to save the output .kmeanspkl file containing kmeans centroids per token.")
    parser.add_argument('--k', type=int, default=20,
                        help="Number of clusters for KMeans.")
    args = parser.parse_args()

    with open(args.input_path, 'rb') as handle:
        print("Load data from", args.input_path)
        data = pickle.load(handle)

    output_key_data = []
    output_dict_data = OrderedDict()
    for key, value in tqdm(data.items()):
        current_k = args.k
        if value.shape[0] <= current_k:
            output_key_data += [key] * value.shape[0]
            output_dict_data[key] = torch.tensor(value).to(torch.bfloat16)
        else:
            kmeans = KMeans(n_clusters=current_k, random_state=0, n_init="auto").fit(value)
            cluster_centers = kmeans.cluster_centers_
            clustered_tensor = torch.tensor(cluster_centers).to(torch.bfloat16)
            output_key_data += [key] * current_k
            output_dict_data[key] = clustered_tensor
    with open(args.output_path, 'wb') as handle:
        pickle.dump(output_dict_data, handle)
    print("Output saved to:", args.output_path)



if __name__ == "__main__":
    main()