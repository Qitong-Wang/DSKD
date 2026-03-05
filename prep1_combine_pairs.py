import json
import argparse
from collections import defaultdict

def load_json(path: str) -> dict[str, set[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # convert lists to sets
    return {k: set(v) for k, v in data.items()}

def merge_dicts(dict1: dict[str, set[str]], dict2: dict[str, set[str]]) -> dict[str, list[str]]:
    merged = defaultdict(set)
    for k, v in dict1.items():
        merged[k].update(v)
    for k, v in dict2.items():
        merged[k].update(v)
    # convert back to lists (sorted)
    return {k: sorted(list(v)) for k, v in merged.items()}

def main():
    parser = argparse.ArgumentParser(description="Merge two synonym JSON dictionaries")
    parser.add_argument("--json1", default="./relationship/wiktionary_ant.json", help="Path to first json file")
    parser.add_argument("--json2", default="./relationship/dict2vec_ant.json", help="Path to second json file")
    parser.add_argument("--output_json",default="./relationship/ant.json",  help="Path to output merged json file")
    args = parser.parse_args()

    dict1 = load_json(args.json1)
    dict2 = load_json(args.json2)
    merged = merge_dicts(dict1, dict2)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
