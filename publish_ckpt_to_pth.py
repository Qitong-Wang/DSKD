# publish_ckpt_to_safetensors.py
# Export training ckpt -> standard inference folder (HF-compatible)
# - Avoids transformers.save_pretrained() to bypass DTensor bug
# - Saves weights as model.safetensors to bypass torch<2.6 torch.load restriction (CVE-related)

import os
import torch
from torch import nn
import transformers

from safetensors.torch import save_file

# your project models
from model import SKDLlamaForCausalLM, SKDMistralForCausalLM


# =======================
# DEFAULT CONFIG (EDIT HERE IF NEEDED)
# =======================
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT_PATH = "./ckpt_ready/llama_dskd.ckpt"
OUT_DIR = "./llama-dskd"
MODEL_TYPE = "llama"          # "llama" or "mistral"


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
CKPT_PATH = "./ckpt_ready/mistral_dskd.ckpt"
OUT_DIR = "./mistral-dskd"
MODEL_TYPE = "mistral"          # "llama" or "mistral"
STUDENT_TOTAL_LAYER = 16
DTYPE = "bf16"                # "bf16" | "fp16" | "fp32"

HF_TOKEN = os.getenv("HF_TOKEN")  # optional
# =======================


def _clean_state_dict(sd: dict) -> dict:
    """Remove common wrappers like DDP 'module.' prefix."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        out[k] = v
    return out


def export_student():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1) load training ckpt ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError(f"ckpt missing 'model_state_dict', keys={list(ckpt.keys())}")

    sd = _clean_state_dict(ckpt["model_state_dict"])

    # ---- 2) build model from base ----
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[DTYPE]

    if MODEL_TYPE == "llama":
        model = SKDLlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float16,  # build lightweight; we cast later
            token=HF_TOKEN,
            trust_remote_code=True,
        )
    elif MODEL_TYPE == "mistral":
        model = SKDMistralForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float16,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
    else:
        raise ValueError("MODEL_TYPE must be 'llama' or 'mistral'")

    # ---- 3) prune layers to student depth ----
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError("Expected model.model.layers to exist (HF-style decoder).")

    model.model.layers = nn.ModuleList(model.model.layers[:STUDENT_TOTAL_LAYER])

    # IMPORTANT: fix config so later from_pretrained builds correct depth
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = STUDENT_TOTAL_LAYER

    # ---- 4) load weights ----
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:3] or unexpected[:3]:
        print("[DEBUG] sample missing:", missing[:3])
        print("[DEBUG] sample unexpected:", unexpected[:3])

    # ---- 5) manual save (NO save_pretrained) ----
    model = model.to(dtype=torch_dtype)

    # 5.1 weights -> safetensors (Transformers will prefer this and avoid torch.load)
    weights_path = os.path.join(OUT_DIR, "model.safetensors")
    state_to_save = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state_to_save, weights_path)

    # (optional) remove old pytorch_model.bin to avoid confusion
    old_bin = os.path.join(OUT_DIR, "pytorch_model.bin")
    if os.path.exists(old_bin):
        os.remove(old_bin)

    # 5.2 config.json
    model.config.save_pretrained(OUT_DIR)

    # 5.3 tokenizer (recommended even if unchanged)
    tok = transformers.AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    tok.save_pretrained(OUT_DIR)

    print("✅ Export finished")
    print("Output dir:", OUT_DIR)
    print("Files:")
    print("  - model.safetensors")
    print("  - config.json")
    print("  - tokenizer files")


if __name__ == "__main__":
    export_student()
