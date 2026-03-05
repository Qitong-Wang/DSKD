import argparse
import json
import pickle
from collections import defaultdict
from transformers import AutoTokenizer


def is_single_token(tokenizer, word: str) -> bool:
    toks = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    return len(toks) == 1


def tokenize_span(tokenizer, word: str):
    return tokenizer(" " + word, add_special_tokens=False)["input_ids"]


def build_m_map(
    tokenizer,
    rel_dict,
    mmax: int,
):

    out_map = defaultdict(list)
    for head, tails in rel_dict.items():
        if not is_single_token(tokenizer, head):
            continue
        if len(head) <= 2:
            continue

        if not is_single_token(tokenizer, head):
            continue
        
        head_tok = tokenize_span(tokenizer, head)[0]

        for tail in tails:
            tail_toks = tokenize_span(tokenizer, tail)

            if len(tail_toks) == 0 or len(tail_toks) > mmax:
                continue

            out_map[head_tok].append(tuple(tail_toks))


    return out_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    ap.add_argument("--mmax", type=int, default=3,
                    help="Max token span length to keep as a span key. If word tokenizes longer, fall back to a representative single token via MorphoLEX piece (if available).")

    ap.add_argument("--in_syn_json", type=str,  default="./relationship/syn_enhanced.json",
                    help="Path to synonyms JSON: {word: [list]}")
    ap.add_argument("--in_ant_json", type=str, default="./relationship/ant_enhanced.json",
                    help="Path to antonyms JSON: {word: [list]}")

    ap.add_argument("--out_syn_pkl", type=str, default="./relationship/mistral_base_syn.pkl",
                    help="Output pickle path for synonym span map Dict[Tuple[int,...], Set[Tuple[int,...]]]")
    ap.add_argument("--out_ant_pkl", type=str, default="./relationship/mistral_base_ant.pkl",
                    help="Output pickle path for antonym span map Dict[Tuple[int,...], Set[Tuple[int,...]]]")

    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    with open(args.in_syn_json, "r", encoding="utf-8") as f:
        syn_dict = json.load(f)

    with open(args.in_ant_json, "r", encoding="utf-8") as f:
        ant_dict = json.load(f)

    syn_map = build_m_map(
        tokenizer,
        syn_dict,
        args.mmax,
    )

    ant_map = build_m_map(
        tokenizer,
        ant_dict,
        args.mmax,
    )

    with open(args.out_syn_pkl, "wb") as f:
        pickle.dump(dict(syn_map), f)

    with open(args.out_ant_pkl, "wb") as f:
        pickle.dump(dict(ant_map), f)

    print("=== DONE ===")


if __name__ == "__main__":
    main()
