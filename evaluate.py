# evaluate.py
from collections import defaultdict
import glob
import os, json, pickle, random, numpy as np, torch, re, string
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers.pipelines.text_generation import Chat, ReturnType
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import SquadDataset, WikiDataset, MultiQADataset, MMLUTestDataset, CoQADataset
from model import SKDLlamaForCausalLM, SKDMistralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- small utils ----------
def my_collate_fn(batch):
    prompts = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    dataset_name = [item[2] for item in batch]
    is_impossible = [item[3] for item in batch]
    return (prompts, answers,dataset_name, is_impossible)

def normalize_text(s):
    def remove_articles(t): return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t): return ' '.join(t.split())
    def remove_punc(t): return ''.join(ch for ch in t if ch not in string.punctuation)
    def lower(t): return t.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(pred_tokens, truth_tokens):
    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(w), truth_tokens.count(w)) for w in common)
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def score_squad(prediction, answers, is_impossible):

    pred_norm = normalize_text(prediction)
    pred_tokens = pred_norm.split()
    norm_answers = [normalize_text(ans) for ans in answers]
    token_answers = [ans.split() for ans in norm_answers]
    if is_impossible:
        return 1.0 if pred_norm.lower() in {"no answer", "unanswerable"} else 0.0
    f1_scores = [compute_f1(pred_tokens, toks) for toks in token_answers]
    return max(f1_scores)
   
def score_coqa(prediction, answers, is_impossible):

    pred_norm = normalize_text(prediction)
    pred_tokens = pred_norm.split()
    norm_answers = [normalize_text(ans) for ans in answers]
    token_answers = [ans.split() for ans in norm_answers]
    if is_impossible:
        return 1.0 if pred_norm.lower() in {"unknown"} else 0.0
    f1_scores = [compute_f1(pred_tokens, toks) for toks in token_answers]
    return max(f1_scores)
   
def get_response(model, message, tokenizer, use_chat=True, max_new_tokens=10, run_option="original"):
    if use_chat:
        msgs = Chat(message)
        model_string = tokenizer.apply_chat_template(
            msgs.messages, truncation=None, padding=False, max_length=None,
            add_generation_prompt=True, tokenize=False
        )

        model_inputs = tokenizer.apply_chat_template(
            msgs.messages, truncation=None, padding=False, max_length=None,
            add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
    else:
        msgs = message
        model_inputs = tokenizer(message, return_tensors='pt')

    model_inputs = model_inputs.to(model.device)

    input_ids = model_inputs["input_ids"]
    attn = model_inputs.get("attention_mask", None)
    in_b = 1 if input_ids.shape[1] == 0 else input_ids.shape[0]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    PAD_ID = tokenizer.pad_token_id
    EOS_ID = tokenizer.eos_token_id
 
    # generate or generate_with_info depending on mode
    if run_option in {"gather"}:
        if isinstance(model, DDP):
            seq, extra = model.module.generate_with_info(
                input_ids=input_ids, attention_mask=attn, pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=max_new_tokens, use_cache=True
            )
        else:
            seq, extra = model.generate_with_info(
                input_ids=input_ids, attention_mask=attn,pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=max_new_tokens, use_cache=True
            )

        # unify extra_info for cluster modes
        if run_option in {"gather"}:
            extra['token_id'] = torch.cat(extra['token_id']).tolist()
            extra['hidden_state'] = np.concatenate(extra['hidden_state'], axis=0)

    else:
        if isinstance(model, DDP):
            seq = model.module.generate(input_ids=input_ids, attention_mask=attn, pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=max_new_tokens,  use_cache=True)
        else:
            seq = model.generate(input_ids=input_ids, attention_mask=attn, pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=max_new_tokens, use_cache=True)

    # postprocess
    out_b = seq.shape[0]
    seq = seq.reshape(in_b, out_b // in_b, *seq.shape[1:])
    seq = seq[0].cpu().numpy().tolist()

    records = []
    for s in seq:
        text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if input_ids is None:
            prompt_len = 0
        else:
            prompt_len = len(tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        all_text = text[prompt_len:]
        if use_chat and isinstance(msgs, Chat):
            all_text = msgs.messages + [{"role": "assistant", "content": all_text}]
        records.append({"generated_text": all_text})

    predicted_answer = text[prompt_len:]
  

    if run_option in {"gather"}:
        return records[0]["generated_text"], predicted_answer, extra
    else:
        return records[0]["generated_text"], predicted_answer

def evaluate(args):

    # DDP
    rank = int(os.getenv("RANK", "0"))
    world_size = args.world_size
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    local_rank = rank % 6
    device = torch.device(f"cuda:{local_rank}")

    # Model setup
    if args.model_name == "llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = SKDLlamaForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16, token=HF_TOKEN
        )
    elif args.model_name == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        model = SKDMistralForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",        
            torch_dtype=torch.float16, 
        )


    print(model)
    training_args = {
        "run_option": args.run_option,
        "loss_type": args.loss_type,
    }

    model.set_training_args(training_args)  

    model.to(device=device, dtype=torch.bfloat16)
    model.model.local_rank = local_rank

    # Load student weights if evaluating a pruned student
    if args.run_option == "student":
        model.model.layers = nn.ModuleList(model.model.layers[:int(args.student_total_layer)])
        ckpt = torch.load(args.load_student_ckpt_path, map_location="cpu")
        new_state = {}
        for k, v in ckpt['model_state_dict'].items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        model.load_state_dict(new_state)

    for p in model.parameters():
        p.requires_grad = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)

    # Dataset selection & generation config
    if "mmlu" in args.task_name and (args.run_option in {"original", "student","teacher_replace"}):
        evaluate_mmlu(args, rank, world_size, model, tokenizer, use_chat=True, max_new=1,  device=device)
        return
    if "wiki" in args.task_name:
        dataset = WikiDataset(args, args.eval_json_path)
        use_chat, max_new = False, 1
    elif any(k in args.task_name for k in ["csqa", "piqa", "winogrande", "arc","mmlu"]):
        name, split = args.task_name.split("_")
        dataset = MultiQADataset(args=args,dataset=name, split=split)
        print("model_name:", args.model_name, "task_name:", args.task_name)
        if args.model_name == "mistral" and  any(k in args.task_name for k in [ "piqa", "winogrande"]):
            use_chat, max_new = True, 2
        else:
            use_chat, max_new = True, 1
    elif "coqa" in args.task_name:
        dataset = CoQADataset(args, split="validation")
        use_chat, max_new = True, 10
    else:
        dataset = SquadDataset(args, args.eval_json_path)
        use_chat, max_new = True, 10
    print("max_new:", max_new)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=my_collate_fn)

    # Mode-specific collectors
    embedding_dict = {}         # gather
    embedding_count = {}        # gather




    total_q, total_score =  0.0, 0.0

    for _, batch in tqdm(enumerate(loader)):
        message, correct_answer, extra_info = batch[0][0], batch[1][0], batch[3][0]

        
        out = get_response(
            model, message, tokenizer, use_chat=use_chat, max_new_tokens=max_new, run_option=args.run_option
        )

        if args.run_option in {"gather"}:
            _, predicted_answer, extra = out
        else:
            _, predicted_answer = out

        # collectors
        if args.run_option == "gather":
            token_id = extra['token_id']
            hidden_state = extra['hidden_state']
            for i in range(len(token_id)):
                tid = int(token_id[i])
                embedding_count[tid] = embedding_count.get(tid, 0) + 1
                embedding_dict.setdefault(tid, [])
                if len(embedding_dict[tid]) < args.gather_size:
                    embedding_dict[tid].append(hidden_state[i])
                else:
                    r = random.randint(1, embedding_count[tid])
                    if r <= args.gather_size:
                        embedding_dict[tid][r-1] = hidden_state[i]

        # scoring
        if "squad" in args.task_name:
            step_score = score_squad(predicted_answer, correct_answer, extra_info)
            print(f"Pred: [{predicted_answer}] | Gold: [{correct_answer}]  | F1: {step_score}")
        elif "coqa" in args.task_name:
            step_score = score_coqa(predicted_answer, correct_answer, extra_info)
            print(f"Pred: [{predicted_answer}] | Gold: [{correct_answer}]  | F1: {step_score}")
        else:
            pred = predicted_answer.strip() 
            if predicted_answer.strip() == correct_answer:
                step_score = 1.0
            else:
                step_score = 0.0
            print(f"Pred: [{predicted_answer}] | Gold: [{correct_answer}]  | Acc: {step_score}")
        total_q += 1
       
        total_score += step_score

    # DDP reduce
    total_tensor = torch.tensor(total_q, dtype=torch.float32, device=device)
    total_score_tensor = torch.tensor(total_score, dtype=torch.float32, device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_score_tensor, op=dist.ReduceOp.SUM)

    avg_f1 = total_score_tensor.item() / total_tensor.item() * 100.0
    if rank == 0:
        if "squad" in args.task_name or "coqa" in args.task_name:
            print(f"All Questions: {int(total_tensor.item())}  | F1 sum: {total_score_tensor.item():.2f}")
            print(f"Result F1: {avg_f1:6.2f}%")
        else:
            print(f"All Questions: {int(total_tensor.item())}  | Acc sum: {total_score_tensor.item():.2f}")
            print(f"Result Acc: {avg_f1:6.2f}%")

    # persist mode-specific artifacts
    if args.run_option == "gather":
        for k, v in embedding_dict.items():
            embedding_dict[k] = np.stack(v)
        with open(os.path.join(args.save_dir, f"{args.task_name}{local_rank}.pkl"), "wb") as f:
            pickle.dump(embedding_dict, f)


def evaluate_mmlu(args,rank, world_size, model, tokenizer, use_chat, max_new, device):

    
    # Load category mapping
    with open("./dataset/MMLU/mmlu_category.json", "r", encoding="utf-8") as f:
        category_info = json.load(f)  

    # Initialize stats
    acc_per_dataset = {}
    acc_by_category = defaultdict(list)
    # ----------------------------------------------------------
    # iterate over every *_test.json → *_dev.json pair
    # ----------------------------------------------------------


    if rank == 0:
        print(">>> Evaluating every dataset in ./test/ and ./dev/ ...")

    acc_per_dataset = {}
    acc_by_category = defaultdict(list)



    if os.path.isdir("./dataset/MMLU/test_json/"): # Check whole MMLU directory
        for test_path in sorted(glob.glob("./dataset/MMLU/test_json/"+"/*_test.json")):
            stem = os.path.basename(test_path).replace("_test.json", "")
            dev_path =  "./dataset/MMLU/dev_json/" + f"{stem}_dev.json"

            # skip quietly if the matching dev file is missing
            if not os.path.exists(dev_path):
                if rank == 0:
                    print(f"[WARN] {dev_path} not found – skipped")
                continue

            total_questions, correct_questions = 0, 0
            dataset = MMLUTestDataset(args, test_path, dev_path)
            sampler = DistributedSampler(dataset,
                                        num_replicas=world_size,
                                        rank=rank,
                                        shuffle=False)
            loader = DataLoader(dataset,
                                batch_size=1,
                                sampler=sampler,
                                collate_fn=my_collate_fn)

            for _, batch in enumerate(loader):
                message = batch[0][0]
                correct_answer = batch[1][0]
              

                out = get_response(
                    model, message, tokenizer, use_chat=use_chat, max_new_tokens=max_new, run_option=args.run_option
                )   
                _, predicted_answer = out

                if predicted_answer.strip() == correct_answer:
                    correct_questions += 1
                total_questions += 1

            # sync across ranks
            correct_tensor = torch.tensor(correct_questions, dtype=torch.float32, device=device)
            total_tensor   = torch.tensor(total_questions,   dtype=torch.float32, device=device)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor,   op=dist.ReduceOp.SUM)

            acc = correct_tensor.item() / total_tensor.item() * 100.0
            if rank == 0:
                acc_per_dataset[stem] = acc

                # Get category
                category = category_info.get(stem, {}).get("category", "Unknown")
                acc_by_category[category].append(acc)
                print(f"{stem:>30s}: {acc:6.2f}%")

        # Final summary
        if rank == 0 and acc_per_dataset:
            print("\n==================== PER-DATASET RESULTS ====================")
            for name, acc in acc_per_dataset.items():
                print(f"{name:>30s}: {acc:6.2f}%")

            print("\n==================== PER-CATEGORY RESULTS ===================")
            macro_total = 0
            for category, accs in acc_by_category.items():
                macro = sum(accs) / len(accs)
                macro_total += macro
                print(f"{category:>20s}: {macro:6.2f}% (from {len(accs):2d} sets)")

            #macro_avg = macro_total / len(acc_by_category)
            print("-------------------------------------------------------------")
            print(f"Macro Average over {len(acc_per_dataset)} datasets: {sum(acc_per_dataset.values()) / len(acc_per_dataset):.2f}%")
            print("=============================================================\n")