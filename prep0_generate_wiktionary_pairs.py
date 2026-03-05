import json
import argparse
import os
from collections import defaultdict
from typing import Tuple, Dict, Set, Iterable, Any

def clean_word(w: str | None) -> str | None:
    if not w:
        return None
    if "#" in w:
        w = w.split("#", 1)[0]
    w = w.strip()
    if not w or w == ";":
        return None
    return w

def add_pair_bidirectional(m: Dict[str, Set[str]], a: str, b: str) -> None:
    #Add (a -> b) and (b -> a) to m. Keep self-loops.
    if a is None or b is None:
        return
    m[a].add(b)
    m[b].add(a)

def iter_rels(entry: Dict[str, Any], rel_key: str) -> Iterable[Tuple[str, str]]:
    head = entry.get("word")
    if not head:
        return
    for sense in entry.get("senses", []):
        for rel in sense.get(rel_key, []):
            w = clean_word(rel.get("word"))
            if not w:
                continue
            yield head, w

def build_maps(jsonl_path: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    syn: Dict[str, Set[str]] = defaultdict(set)
    ant: Dict[str, Set[str]] = defaultdict(set)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("lang") != "English":
                continue

            for a, b in iter_rels(entry, "synonyms"):
                add_pair_bidirectional(syn, a, b)

            for a, b in iter_rels(entry, "antonyms"):
                add_pair_bidirectional(ant, a, b)

    syn_out = {k: sorted(v) for k, v in syn.items() if v}
    ant_out = {k: sorted(v) for k, v in ant.items() if v}
    return syn_out, ant_out

def main():
    ap = argparse.ArgumentParser(
        description="Extract English synonym & antonym pairs from Kaikki/Wiktextract JSONL."
    )
    ap.add_argument(
        "--jsonl",
        default="./data_source/kaikki.org-dictionary-English.jsonl",
        help="Path to Kaikki JSONL file",
    )
    ap.add_argument(
        "--out-dir",
        default="./relationship",
        help="Directory to write output JSON files (default: current directory)."
    )
    args = ap.parse_args()

    syn, ant = build_maps(args.jsonl)

    os.makedirs(args.out_dir, exist_ok=True)
    syn_path = os.path.join(args.out_dir, "english_syn.json")
    ant_path = os.path.join(args.out_dir, "english_ant.json")

    with open(syn_path, "w", encoding="utf-8") as f:
        json.dump(syn, f, ensure_ascii=False, indent=2)
    with open(ant_path, "w", encoding="utf-8") as f:
        json.dump(ant, f, ensure_ascii=False, indent=2)

    print(f" Wrote {syn_path} and {ant_path}")

if __name__ == "__main__":
    main()
