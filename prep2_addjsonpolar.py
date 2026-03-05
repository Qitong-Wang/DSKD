import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


DEFAULT_NEG_PREFIXES = {
    "un", "dis", "non",
    "in", "im", "il", "ir",
    "anti", "de", "a", "an"
}
DEFAULT_NEG_SUFFIXES = {
    "less", "free",
}

EXTRA_NEG_PREFIXES = {"anti", "de", "a", "an"}
EXTRA_NEG_SUFFIXES = {"free",}


def parse_morpholex_segm(seg: str) -> List[Tuple[str, str]]:
    if seg is None:
        return []
    s = str(seg)

    pieces: List[Tuple[str, str]] = []
    i = 0
    n = len(s)

    def _clean_piece(x: str) -> str:
        x2 = x.strip().lower()
        x2 = re.sub(r"[^a-z0-9\-']", "", x2)
        return x2

    while i < n:
        ch = s[i]
        if ch == "<":
            j = s.find("<", i + 1)
            if j == -1:
                break
            raw = s[i + 1: j]
            piece = _clean_piece(raw)
            if piece:
                pieces.append(("pref", piece))
            i = j + 1
            continue

        if ch == "(":
            j = s.find(")", i + 1)
            if j == -1:
                break
            raw = s[i + 1: j]
            piece = _clean_piece(raw)
            if piece:
                pieces.append(("root", piece))
            i = j + 1
            continue

        if ch == ">":
            j = s.find(">", i + 1)
            if j == -1:
                break
            raw = s[i + 1: j]
            piece = _clean_piece(raw)
            if piece:
                pieces.append(("suf", piece))
            i = j + 1
            continue

        i += 1

    return pieces


def build_morph_table_from_xlsx(xlsx_path: str) -> Dict[str, Dict]:
    import pandas as pd

    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    ignore = {"Presentation", "All prefixes", "All suffixes", "All roots"}

    morph: Dict[str, Dict] = {}

    for sheet in xl.sheet_names:
        if sheet in ignore:
            continue

        df = pd.read_excel(
            xlsx_path,
            sheet_name=sheet,
            usecols=["Word", "MorphoLexSegm"],
            engine="openpyxl",
        )
        for w, seg in zip(df["Word"].astype(str), df["MorphoLexSegm"].astype(str)):
            word = w.strip().lower()
            if not word or word == "nan":
                continue

            pieces = parse_morpholex_segm(seg)

            if word not in morph:
                morph[word] = {"segm": seg, "pieces": pieces}
            else:
                seen = set(morph[word]["pieces"])
                for p in pieces:
                    if p not in seen:
                        morph[word]["pieces"].append(p)
                        seen.add(p)

    return morph


def reconstruct_base_and_polarity(
    word: str,
    morph: Dict[str, Dict],
    neg_prefixes: Set[str],
    neg_suffixes: Set[str],
    require_base_in: str,
    vocab: Set[str],
) -> Tuple[int, Optional[str], List[Tuple[str, str]], List[Tuple[str, str]], int]:
    w = word.strip().lower()
    entry = morph.get(w)
    if not entry:
        return (1, None, [], [], 0)

    pieces: List[Tuple[str, str]] = entry["pieces"]

    removed: List[Tuple[str, str]] = []
    kept: List[Tuple[str, str]] = []

    for ptype, piece in pieces:
        if ptype == "pref" and piece in neg_prefixes:
            removed.append((ptype, piece))
        elif ptype == "suf" and piece in neg_suffixes:
            removed.append((ptype, piece))
        else:
            kept.append((ptype, piece))

    neg_count = len(removed)
    base = "".join(piece for _, piece in kept).strip()
    if not base or base == w:
        base_candidate = None
    else:
        base_candidate = base

    def base_exists(b: Optional[str]) -> bool:
        if not b:
            return False
        if require_base_in == "none":
            return True
        if require_base_in == "morph":
            return b in morph
        if require_base_in == "vocab":
            return b in vocab
        if require_base_in == "morph_or_vocab":
            return (b in morph) or (b in vocab)
        return (b in morph) or (b in vocab)

    if neg_count % 2 == 1 and base_exists(base_candidate):
        return (-1, base_candidate, pieces, removed, neg_count)

    return (1, base_candidate, pieces, removed, neg_count)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dedup_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morpholex_xlsx", default="./MorphoLEX_en.xlsx", help="Path to MorphoLEX xlsx")
    ap.add_argument("--in_syn_json", type=str, default="./relationship/syn.json",
                    help="Path to synonyms JSON: {word: [list]}")
    ap.add_argument("--in_ant_json", type=str, default="./relationship/ant.json",
                    help="Path to antonyms JSON: {word: [list]}")

    ap.add_argument("--out_syn_json", type=str, default="./relationship/syn_enhanced.json",
                    help="Path to output synonyms JSON: {word: [list]}")
    ap.add_argument("--out_ant_json", type=str, default="./relationship/ant_enhanced.json",
                    help="Path to output antonyms JSON: {word: [list]}")

    ap.add_argument("--polarity_analysis_tsv", type=str, default="./relationship/polarity_analysis.tsv",
                    )
    ap.add_argument("--flip_analysis_tsv", type=str, default="./relationship/flip_analysis.tsv",
                    )

    ap.add_argument("--require_base_in", default="morph_or_vocab",
                    choices=["morph", "vocab", "morph_or_vocab", "none"],
                    help="How strict to be about base_candidate existence before allowing polarity=-1.")

    ap.add_argument("--use_extra_affixes", action="store_true",
                    help="Enable extra, riskier neg-affixes (anti/de/a/an, free/proof). Start with False.")

    args = ap.parse_args()

    syn = load_json(args.in_syn_json)
    ant = load_json(args.in_ant_json)

    vocab: Set[str] = set()
    for h, tails in syn.items():
        vocab.add(str(h).strip().lower())
        for t in tails:
            vocab.add(str(t).strip().lower())
    for h, tails in ant.items():
        vocab.add(str(h).strip().lower())
        for t in tails:
            vocab.add(str(t).strip().lower())

    morph = build_morph_table_from_xlsx(args.morpholex_xlsx)

    neg_prefixes = set(DEFAULT_NEG_PREFIXES)
    neg_suffixes = set(DEFAULT_NEG_SUFFIXES)
    if args.use_extra_affixes:
        neg_prefixes |= EXTRA_NEG_PREFIXES
        neg_suffixes |= EXTRA_NEG_SUFFIXES

    pol_cache: Dict[str, Tuple[int, Optional[str], List[Tuple[str, str]], List[Tuple[str, str]], int]] = {}
    polarity_rows: List[str] = []
    polarity_rows.append(
        "word\tin_morph\tsegm\tpieces\tremoved_neg_pieces\tneg_count\tbase_candidate\tbase_in_morph\tbase_in_vocab\tpolarity\n"
    )

    def get_pol_info(word: str):
        w = str(word).strip().lower()
        if w in pol_cache:
            return pol_cache[w]

        pol, base, pieces, removed, neg_count = reconstruct_base_and_polarity(
            w, morph, neg_prefixes, neg_suffixes, args.require_base_in, vocab
        )
        pol_cache[w] = (pol, base, pieces, removed, neg_count)

        in_morph = 1 if w in morph else 0
        segm = morph[w]["segm"] if w in morph else ""
        pieces_str = "|".join([f"{pt}:{pc}" for pt, pc in pieces])
        removed_str = "|".join([f"{pt}:{pc}" for pt, pc in removed])

        base_in_morph = 1 if (base and base in morph) else 0
        base_in_vocab = 1 if (base and base in vocab) else 0

        polarity_rows.append(
            f"{w}\t{in_morph}\t{segm}\t{pieces_str}\t{removed_str}\t{neg_count}\t{base or ''}\t{base_in_morph}\t{base_in_vocab}\t{pol}\n"
        )
        return pol_cache[w]

    # normalize word to base when pol=-1
    def norm_word(original: str, info) -> str:
        pol, base, _, _, _ = info
        if pol == -1 and base:
            return base
        return original

    syn_out = defaultdict(list)
    ant_out = defaultdict(list)

    flip_rows: List[str] = []
    flip_rows.append(
        "source\tmoved_to\thead\ttail\tpol_head\tpol_tail\tparity\tfinal_sign\thead_base\ttail_base\thead_removed_neg\ttail_removed_neg\thead_out\ttail_out\n"
    )

    stats = {
        "syn_pairs_in": 0,
        "ant_pairs_in": 0,
        "syn_keep": 0,
        "syn_flip_to_ant": 0,
        "ant_keep": 0,
        "ant_flip_to_syn": 0,
        "drop_self_in_syn": 0,
    }

    def record_flip(source: str, moved_to: str, h: str, t: str, hinfo, tinfo, parity: int, final_sign: int, hout: str, tout: str):
        hpol, hbase, _, hremoved, _ = hinfo
        tpol, tbase, _, tremoved, _ = tinfo
        hrem = "|".join([f"{pt}:{pc}" for pt, pc in hremoved])
        trem = "|".join([f"{pt}:{pc}" for pt, pc in tremoved])
        flip_rows.append(
            f"{source}\t{moved_to}\t{h}\t{t}\t{hpol}\t{tpol}\t{parity}\t{final_sign}\t{hbase or ''}\t{tbase or ''}\t{hrem}\t{trem}\t{hout}\t{tout}\n"
        )

    # Process syn source (syn=+1)
    REL_SIGN_SYN = 1
    for head, tails in syn.items():
        h = str(head).strip()
        hinfo = get_pol_info(h)
        hpol = hinfo[0]
        for tail in tails:
            t = str(tail).strip()
            tinfo = get_pol_info(t)
            tpol = tinfo[0]

            stats["syn_pairs_in"] += 1

            parity = hpol * tpol
            final_sign = REL_SIGN_SYN * parity  # +1 => syn, -1 => ant

            # normalize to base if negated
            hout = norm_word(h, hinfo)
            tout = norm_word(t, tinfo)

            # syn cannot contain itself (after normalization too)
            if final_sign > 0 and hout.strip().lower() == tout.strip().lower():
                stats["drop_self_in_syn"] += 1
                continue

            if final_sign > 0:
                syn_out[hout].append(tout)
                stats["syn_keep"] += 1
            else:
                ant_out[hout].append(tout)
                stats["syn_flip_to_ant"] += 1
                record_flip("syn", "ant", h, t, hinfo, tinfo, parity, final_sign, hout, tout)

    # Process ant source (ant=-1)
    REL_SIGN_ANT = -1
    for head, tails in ant.items():
        h = str(head).strip()
        hinfo = get_pol_info(h)
        hpol = hinfo[0]
        for tail in tails:
            t = str(tail).strip()
            tinfo = get_pol_info(t)
            tpol = tinfo[0]

            stats["ant_pairs_in"] += 1

            parity = hpol * tpol
            final_sign = REL_SIGN_ANT * parity  # +1 => syn, -1 => ant

            # normalize to base if negated
            hout = norm_word(h, hinfo)
            tout = norm_word(t, tinfo)

            # syn cannot contain itself (after normalization too)
            if final_sign > 0 and hout.strip().lower() == tout.strip().lower():
                stats["drop_self_in_syn"] += 1
                continue

            if final_sign > 0:
                syn_out[hout].append(tout)
                stats["ant_flip_to_syn"] += 1
                record_flip("ant", "syn", h, t, hinfo, tinfo, parity, final_sign, hout, tout)
            else:
                ant_out[hout].append(tout)
                stats["ant_keep"] += 1

    # Dedup + final guard
    syn_out = {h: dedup_preserve([t for t in ts if t.strip().lower() != h.strip().lower()]) for h, ts in syn_out.items() if ts}
    ant_out = {h: dedup_preserve(ts) for h, ts in ant_out.items() if ts}

    save_json(syn_out, args.out_syn_json)
    save_json(ant_out, args.out_ant_json)

    with open(args.polarity_analysis_tsv, "w", encoding="utf-8") as f:
        f.writelines(polarity_rows)

    with open(args.flip_analysis_tsv, "w", encoding="utf-8") as f:
        f.writelines(flip_rows)

    print("=== DONE ===")
    print("Saved syn_pol:", args.out_syn_json)
    print("Saved ant_pol:", args.out_ant_json)
    print("Saved polarity_analysis:", args.polarity_analysis_tsv)
    print("Saved flip_analysis:", args.flip_analysis_tsv)
    print("Stats:", stats)


if __name__ == "__main__":
    main()
