import argparse
import pickle
from typing import Dict, Set, Tuple, Optional, Any

import torch
from tqdm import tqdm

Span = Tuple[int, ...]

def normalize_head_maps(raw_map: Any) -> Dict[int, Set[Span]]:
    out: Dict[int, Set[Span]] = {}
    for k, vs in raw_map.items():
        if isinstance(k, tuple):
            if len(k) != 1:
                continue
            head = int(k[0])
        else:
            head = int(k)

        sset: Set[Span] = set()
        for v in vs:
            sset.add(tuple(int(x) for x in v))
        if sset:
            out[head] = sset
    return out


@torch.no_grad()
def _pair_nn_l2_raw(Aq: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    #For each row in Aq, find raw-L2 nearest neighbor in B; return matched rows from B.
    if Aq.dim() != 2 or B.dim() != 2:
        raise ValueError("Aq and B must be 2D")
    if Aq.numel() == 0 or B.numel() == 0:
        return Aq.new_zeros((Aq.size(0), Aq.size(1)))
    dist = torch.cdist(Aq.to(torch.float32), B.to(torch.float32))  # (K, nB)
    ib = torch.argmin(dist, dim=1)
    return B.index_select(0, ib)


def _sample_K_rows(A: torch.Tensor, K: int) -> torch.Tensor:
    #Sample up to K rows from A (without replacement). Return (K', D).

    if A.dim() != 2:
        raise ValueError("A must be 2D")
    nA, d = int(A.size(0)), int(A.size(1))
    if nA <= 0:
        return A.new_zeros((0, d))
    K = min(int(K), nA)
    if K <= 0:
        return A.new_zeros((0, d))
    if K == nA:
        idx = torch.arange(nA, device=A.device)
    else:
        idx = torch.randperm(nA, device=A.device)[:K]
    return A.index_select(0, idx)


@torch.no_grad()
def span_cloud_itersum_equal(span: Span, sense_dict: Dict[int, torch.Tensor], K: int, m_max: int):
    L = len(span)
    if L == 0:
        return None
    if L == 1:
        return sense_dict.get(int(span[0]))
    if L > m_max:
        return None

    t1 = int(span[0])
    S1 = sense_dict.get(t1)
    if S1 is None or S1.dim()!=2 or S1.numel()==0:
        return None

    cur = _sample_K_rows(S1, K=K)      # (K',D) 作为 aligned(S1)
    if cur.numel()==0:
        return None

    acc = cur.to(torch.float32)        # SUM of aligned clouds
    k_tokens = 1

    for tid in span[1:]:
        B = sense_dict.get(int(tid))
        if B is None or B.dim()!=2 or B.numel()==0:
            return None

        mean_prev = acc / float(k_tokens)          # mean of previous aligned
        aligned = _pair_nn_l2_raw(mean_prev, B)    # match using mean
        acc = acc + aligned.to(torch.float32)
        k_tokens += 1

    out = (acc / float(k_tokens)).to(dtype=S1.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--syn_pkl", type=str, default="./syn_morpho_dict/mistral_syn.pkl", help="Input synonym span map pkl")
    ap.add_argument("--ant_pkl", type=str, default="./syn_morpho_dict/mistral_ant.pkl", help="Input antonym span map pkl")
    ap.add_argument("--sense_pkl", type=str)


    ap.add_argument("--out_syn_vid", type=str)
    ap.add_argument("--out_ant_vid", type=str)
    ap.add_argument("--out_cloud_dict", type=str)
    ap.add_argument("--out_span2vid", type=str, default="")

    ap.add_argument("--vid_base", type=int, default=200000)
    ap.add_argument("--K", type=int, default=25, help="Number of sampled senses/rows kept for composed span clouds")
    ap.add_argument("--m_max", type=int, default=3, help="Drop spans with len(span) > m_max")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--dtype", type=str, default="bf16",
                    help="dtype for cached clouds on CPU")

    args = ap.parse_args()
    torch.manual_seed(int(args.seed))

    # output dtype
    if args.dtype == "bf16":
        out_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        out_dtype = torch.float16
    else:
        out_dtype = torch.float32

    # load
    with open(args.syn_pkl, "rb") as f:
        syn_raw = pickle.load(f)
    with open(args.ant_pkl, "rb") as f:
        ant_raw = pickle.load(f)
    with open(args.sense_pkl, "rb") as f:
        sense_dict: Dict[int, torch.Tensor] = pickle.load(f)

    syn_map = normalize_head_maps(syn_raw)
    ant_map = normalize_head_maps(ant_raw)

    print(f"heads in syn_map: {len(syn_map)}")
    print(f"heads in ant_map: {len(ant_map)}")
    print(f"sense_dict token clouds: {len(sense_dict)}")

    # unified cloud dict: tid -> token cloud, vid -> span cloud
    cloud_dict: Dict[int, torch.Tensor] = {}

    # token clouds under tid (always keep)
    for tid, mat in sense_dict.items():
        if isinstance(mat, torch.Tensor) and mat.dim() == 2 and mat.numel() > 0:
            cloud_dict[int(tid)] = mat.to(dtype=out_dtype, device="cpu")

    # build neighbor maps + span2vid (global unique)
    span2vid: Dict[Span, int] = {}
    vid2span: Dict[int, Span] = {}
    next_vid = int(args.vid_base)

    def get_vid_for_span(sp: Span) -> int:
        nonlocal next_vid
        v = span2vid.get(sp)
        if v is None:
            v = next_vid
            next_vid += 1
            span2vid[sp] = v
            vid2span[v] = sp
        return v

    def build_outputs(src: Dict[int, Set[Span]]) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
        out_tid: Dict[int, Set[int]] = {}
        out_vid: Dict[int, Set[int]] = {}

        dropped_too_long = 0
        kept_multi = 0

        for head, spans in src.items():
            tids: Set[int] = set()
            vids: Set[int] = set()

            for sp in spans:
                sp = tuple(int(x) for x in sp)
                L = len(sp)

                if L == 1:
                    tids.add(int(sp[0]))
                    continue

                if L > int(args.m_max):
                    dropped_too_long += 1
                    continue

                # multi-token span kept -> assign VID
                v = get_vid_for_span(sp)
                vids.add(int(v))
                kept_multi += 1

            if tids:
                out_tid[int(head)] = tids
            if vids:
                out_vid[int(head)] = vids

        print(f"dropped multi-token neighbors due to len>m_max: {dropped_too_long}")
        print(f"kept multi-token neighbor edges: {kept_multi}")
        return out_tid, out_vid

    syn_tid, syn_vid = build_outputs(syn_map)
    ant_tid, ant_vid = build_outputs(ant_map)

    # build clouds for VIDs (compose)
    fail_no_sense = 0
    fail_empty = 0
    built = 0

    for v, sp in tqdm(list(vid2span.items()), desc="Build VID span clouds (itersum_equal)"):
        mat = span_cloud_itersum_equal(sp, sense_dict, K=int(args.K), m_max=int(args.m_max))
        if mat is None:
            fail_no_sense += 1
            continue
        if mat.numel() == 0:
            fail_empty += 1
            continue
        cloud_dict[int(v)] = mat.to(dtype=out_dtype, device="cpu")
        built += 1

    print(f"unique VIDs assigned: {len(vid2span)}")
    print(f"VID span clouds built: {built}")
    print(f"VID spans failed (missing sense / dropped / None): {fail_no_sense}")
    print(f"VID spans failed (empty): {fail_empty}")
    print(f"unified cloud_dict size (tid+vid): {len(cloud_dict)}")

    # save
    with open(args.out_syn_vid, "wb") as f:
        pickle.dump(syn_vid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.out_ant_vid, "wb") as f:
        pickle.dump(ant_vid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.out_cloud_dict, "wb") as f:
        pickle.dump(cloud_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    if args.out_span2vid:
        with open(args.out_span2vid, "wb") as f:
            pickle.dump({"span2vid": span2vid, "vid2span": vid2span}, f, protocol=pickle.HIGHEST_PROTOCOL)

 
    print(f"saved syn_vid -> {args.out_syn_vid}")
    print(f"saved ant_vid -> {args.out_ant_vid}")
    print(f"saved unified cloud_dict -> {args.out_cloud_dict}")
    if args.out_span2vid:
        print(f"saved span2vid/vid2span -> {args.out_span2vid}")


if __name__ == "__main__":
    main()