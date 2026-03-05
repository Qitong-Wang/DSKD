import argparse
import json
from collections import defaultdict

def build_synonym_groups(txt_path: str) -> dict[str, list[str]]:
    graph = defaultdict(set)

    # Build undirected graph
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            w1, w2 = parts
            graph[w1].add(w2)
            graph[w2].add(w1)

    # convert sets to lists for JSON
    return {k: sorted(list(v)) for k, v in graph.items()}

def main():
    parser = argparse.ArgumentParser(description="Convert synonym txt file to merged bidirectional JSON dictionary")
    parser.add_argument("--input_txt", default="./data_source/ant-pairs.txt", help="Path to input txt file (pairs of words, space-separated)")
    parser.add_argument("--output_json", default="./relationship/dict2vec_ant.json", help="Path to output json file")
    args = parser.parse_args()

    syn_dict = build_synonym_groups(args.input_txt)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(syn_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
