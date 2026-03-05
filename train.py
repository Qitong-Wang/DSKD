
import math
import os, time, random, pickle
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import get_scheduler
from transformers.pipelines.text_generation import Chat, ReturnType
from transformers import AutoConfig
from dataset import SquadDataset, MultiQADataset
from model import SKDLlamaForCausalLM, SKDMistralForCausalLM
from loss_function import kd_loss_logits, syn_ant_loss_no_norm
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

HF_TOKEN = os.getenv("HF_TOKEN")

def replace_sense_topk(input_emb, sense_tensor_emb, metric="dot", k=1, mode="near"):
    if input_emb.dim() != 1:
        raise ValueError("input_emb must be 1D")
    if sense_tensor_emb.dim() != 2:
        raise ValueError("sense_tensor_emb must be 2D")
    k = min(k, sense_tensor_emb.size(0))
    inp = input_emb.unsqueeze(0)
    largest = (mode == "near")
    if metric == "dot":
        scores = inp @ sense_tensor_emb.T
        _, idx = torch.topk(scores, k=k, dim=1, largest=largest)
    elif metric == "l2":
        d = torch.norm(inp.unsqueeze(1) - sense_tensor_emb.unsqueeze(0), dim=2)
        _, idx = torch.topk(d, k=k, dim=1, largest=not largest)
    elif metric == "cos":
        x = F.normalize(inp, p=2, dim=1)
        y = F.normalize(sense_tensor_emb, p=2, dim=1)
        scores = x @ y.T
        _, idx = torch.topk(scores, k=k, dim=1, largest=largest)
    else:
        raise ValueError("Unsupported metric")
    idx = idx.squeeze(0)
    return idx, sense_tensor_emb[idx]



def my_collate_fn(batch):
    prompts = [item[0] for item in batch]
    answers = [item[1] for item in batch]
    dataset_name = [item[2] for item in batch]
    dataset_extra = [item[3] for item in batch]
    return (prompts, answers, dataset_name, dataset_extra)

def train(args):
    # Load dicts needed for training
    if args.loss_type < 99:
        # VID-only dictionaries
        # syn/ant pkl must be wrapper dicts: {"syn_vid": ...} and {"ant_vid": ...}
        with open(args.syn_dict_path, "rb") as f:
            syn_obj = pickle.load(f)
        with open(args.ant_dict_path, "rb") as f:
            ant_obj = pickle.load(f)


        syn_dict = syn_obj  # head_tid -> set[VID]
        ant_dict = ant_obj  # head_tid -> set[VID]
        syn_ant_keys = set(syn_dict.keys()) | set(ant_dict.keys())

        with open(args.sense_dict_path, "rb") as f:
            sense_dict = pickle.load(f)


    # DDP init
    rank = int(os.getenv("RANK", "0"))
    world_size = args.world_size
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    teacher_start = args.teacher_device_start_indice
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    assert local_rank == (rank % teacher_start), "local_rank mismatch with teacher_device_start_indice"

    teacher_device = torch.device(f"cuda:{local_rank + teacher_start}")
    student_device = torch.device(f"cuda:{local_rank}")

    # Teacher (frozen)
    if args.model_name == "llama":
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        with torch.no_grad():
            teacher_model = SKDLlamaForCausalLM.from_pretrained(
                model_name, device_map="cpu", torch_dtype=torch.float32, token=HF_TOKEN
            )
            teacher_model.to(device=teacher_device, dtype=torch.bfloat16)
            for p in teacher_model.parameters():
                p.requires_grad = False
        teacher_model.eval()
        # Student

        config = AutoConfig.from_pretrained(
            model_name,
            token=HF_TOKEN,
            trust_remote_code=True,
        )
        config.attention_dropout = args.dropout
      
        student_model = SKDLlamaForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32, token=HF_TOKEN,  config=config,
        )
        
    elif args.model_name == "mistral":
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        with torch.no_grad():
            teacher_model = SKDMistralForCausalLM.from_pretrained(
                model_name, device_map="cpu", torch_dtype=torch.float32, token=HF_TOKEN
            )
            teacher_model.to(device=teacher_device, dtype=torch.bfloat16)
            for p in teacher_model.parameters():
                p.requires_grad = False
        teacher_model.eval()
        # Student
        student_model = SKDMistralForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float32, token=HF_TOKEN
        )

    # keep first N layers
    student_model.model.layers = nn.ModuleList(student_model.model.layers[:args.student_total_layer])

    # Freeze bottom (total - train) layers
    for i in range(args.student_total_layer - args.student_train_layer):
        for p in student_model.model.layers[i].parameters():
            p.requires_grad = False


    for p in student_model.model.embed_tokens.parameters():
        p.requires_grad = False
    for p in student_model.lm_head.parameters():
        p.requires_grad = False
    for p in student_model.model.norm.parameters():
        p.requires_grad = False

    if rank == 0:
        print(student_model)
    # Optional checkpoint load
    load_checkpoint = args.load_student_ckpt_path != "empty"
    if load_checkpoint:
        ckpt = torch.load(args.load_student_ckpt_path, map_location="cpu")
        new_state = {}
        for k, v in ckpt['model_state_dict'].items():
            new_state[k[7:]] = v if k.startswith("module.") else v
        student_model.load_state_dict(new_state)
        print(f"Loaded checkpoint from {args.load_student_ckpt_path}, epoch {ckpt.get('epoch', 'N/A')}")
        start_epoch = ckpt.get('epoch', -1) + 1
    else:
        print("No ckpt found, start from scratch")
        start_epoch = 0

    student_model.to(device=student_device, dtype=torch.bfloat16)
    student_model = DDP(student_model, device_ids=[student_device.index], find_unused_parameters=False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)


    # Data
    if args.task_name == "squad":
        train_dataset = SquadDataset(args, args.train_json_path)
    elif args.task_name == "cls":
        specs = ["csqa", "piqa", "arc", "mmlu"]
     
        def downsample_dataset(dataset, num_samples, seed):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
            return Subset(dataset, indices)

        # compute total size across all datasets
        sizes = []
        for name in specs:
            ds_full = MultiQADataset(args=args, dataset=name, split="train")
            sizes.append(len(ds_full))

        total_size = sum(sizes)
        target_size = int(0.25 * total_size)      # 0.2 in Oct #0.25 in Dec

        downsized = []
        for i, (name, ds_len) in enumerate(zip(specs, sizes)):
            ds_full = MultiQADataset(args=args, dataset=name, split="train")
            # scale proportionally
            ds_target = int(target_size * ds_len / total_size)
            ds_small = downsample_dataset(ds_full, num_samples=ds_target, seed=42 + 100*i)
            downsized.append(ds_small)

        train_dataset = ConcatDataset(downsized)
    elif args.task_name == "all":
        specs = ["csqa", "piqa", "arc", "mmlu"]
    
        def downsample_dataset(dataset, num_samples, seed):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
            return Subset(dataset, indices)

        # compute total size across all datasets
        sizes = []
        for name in specs:
            ds_full = MultiQADataset(args=args, dataset=name, split="train")
            sizes.append(len(ds_full))

        total_size = sum(sizes)
        target_size = int(0.25 * total_size)      

        downsized = []
        for i, (name, ds_len) in enumerate(zip(specs, sizes)):
            ds_full = MultiQADataset(args=args, dataset=name, split="train")
            # scale proportionally
            ds_target = int(target_size * ds_len / total_size)
            ds_small = downsample_dataset(ds_full, num_samples=ds_target, seed=42 + 100*i)
            downsized.append(ds_small)
        downsized.append(SquadDataset(args, args.train_json_path))



        train_dataset = ConcatDataset(downsized)
   

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, collate_fn=my_collate_fn)

    # Optim/sched
    lr = args.lr * args.world_size
    optimizer = optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-3)
    if load_checkpoint and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    num_training_steps = max(1, int(len(train_dataset) / world_size / 1 * args.epoch))
    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=max(1, num_training_steps // 10),
        num_training_steps=num_training_steps
    )
    if load_checkpoint and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    loss_tok = nn.CrossEntropyLoss()

    teacher_model.run_option = "train_teacher"
    for epoch in tqdm(range(start_epoch, args.epoch)):
        sampler.set_epoch(epoch)


        for batch_idx, batch in enumerate(tqdm(loader)):
            student_model.train()
            optimizer.zero_grad()

            message = batch[0][0]
            #print("dataset_name", batch[2][0], "dataset_extra_info", batch[3][0])
           
            chats = Chat(message)
            model_inputs = tokenizer.apply_chat_template(
                chats.messages, truncation=None, padding=False, max_length=None,
                add_generation_prompt=True, return_dict=True, return_tensors="pt",
            )
            
            input_ids = model_inputs["input_ids"]
            attention_mask = model_inputs.get("attention_mask", None)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            PAD_ID = tokenizer.pad_token_id
            EOS_ID = tokenizer.eos_token_id

            # Build teacher continuation to create combined_ids
            with torch.no_grad():
                if batch[2][0] == "squad":       
                    gen, extra = teacher_model.generate_with_info(
                        input_ids=input_ids.to(teacher_device),
                        attention_mask=attention_mask.to(teacher_device) if attention_mask is not None else None,
                        pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=10, use_cache=True
                    )

                elif batch[2][0] in ["csqa", "piqa", "arc", "mmlu"]:

                    if batch[2][0] == "piqa" and args.model_name =="mistral":
                        gen, extra = teacher_model.generate_with_info(
                        input_ids=input_ids.to(teacher_device),
                        attention_mask=attention_mask.to(teacher_device) if attention_mask is not None else None,
                        pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=2, use_cache=True
                        )  
                    else:
                        gen, extra = teacher_model.generate_with_info(
                            input_ids=input_ids.to(teacher_device),
                            attention_mask=attention_mask.to(teacher_device) if attention_mask is not None else None,
                            pad_token_id=PAD_ID, eos_token_id=EOS_ID, max_new_tokens=1, use_cache=True
                        )  
                else:
                    raise ValueError("Unsupported dataset name")
                
            teacher_full_token_ids = torch.cat(extra['token_ids']).tolist()
            teacher_full_hidden_states = torch.cat(extra['hidden_states'], axis=0)
            teacher_full_logits = torch.cat(extra['logits'], axis=0)
        

            in_b = input_ids.shape[0]
            out_b = gen.shape[0]
            gen = gen.reshape(in_b, out_b // in_b, *gen.shape[1:])[0]

             #######
            # input_ids.shape = (1, L_in)                                (1,100)
            # token_ids.shape = (L_in + k-1,)                        (1,109)   
            # gen.shape = (1, L_in + k)                              (1,110)
            # gen[ :L_in] = input_ids
            # gen[ L_in:] = token_ids[-k:]                               k = 2
            # k = len(gen) - L_in  (k is number of newly generated tokens)
            # token_ids has one less token than gen, because it is shifted left by 1
            # hidden_states.shape = (L_in + k-1, D), mapped to token_ids  (109, 4096)
           
            gen[:, :input_ids.shape[1]] = input_ids

            L_kd = random.randint(input_ids.shape[1], gen.shape[1]-1)
            combined_ids = gen[:, :L_kd]

            # Forward both
            student_out = student_model(
                input_ids=combined_ids.to(student_device),
                attention_mask=attention_mask, output_hidden_states=True, logits_to_keep=0,
                pad_token_id=PAD_ID, eos_token_id=EOS_ID,
            )

            # original: pad_token_id = 128009
            stud_hidden = student_out.hidden_states[-1]
        
            teach_hidden = teacher_full_hidden_states[:L_kd, :].to(student_device)
            teach_logits = teacher_full_logits[:L_kd, :].to(student_device)
        

            # synonym/antonym pull-push
            total_syn_ant = torch.tensor(0.0, device=student_device)
            sense_tokens = gen[:,1:L_kd+1]   # shifted left by 1 compared to combined_ids
            if args.loss_type < 90:

                assert sense_tokens.shape[1] == teach_hidden.shape[0]
                #tokens = combined_ids[0].tolist()
                sense_ids = sense_tokens[0].tolist()

                # positions where token in syn/ant dicts
                valid = [(i, tid) for i, tid in enumerate(sense_ids) if tid in syn_ant_keys]
                #print("total_length", len(sense_ids), "len syn_ant_keys", len(valid))
                sample_length = min(25,len(valid))  # sample length. usually is sample_size*len(valid). But if len(valid) is small(<25), use len(valid)
                if len(valid)* args.sample_size> 25:
                    sample_length = int(args.sample_size*len(valid))
                pick = random.sample(valid, k=sample_length)

                tmp_losses = []
                for pos, tid in pick:
                    target = stud_hidden[0, pos, :]
                    teacher_target = teach_hidden[pos, :]

                    syn_mat = None
                    if args.neighbors > 0 and tid in syn_dict and syn_dict[tid]:
                        syn_rows = []
                        for vid in random.sample(syn_dict[tid], k=min(args.neighbors, len(syn_dict[tid]))):
                            mat_t = sense_dict.get(int(vid))
                            if mat_t is not None:
                                syn_rows.append(mat_t)
                        if tid in sense_dict:
                            syn_rows.append(sense_dict[tid])
                        if syn_rows:
                            syn_mat = torch.vstack(syn_rows).to(student_device)
                            if args.kappa < 99:
                                syn_mat = replace_sense_topk(teacher_target, syn_mat, "l2",
                                                            args.kappa, mode="near")[1]

                    ant_mat = None
                    if args.neighbors > 0 and tid in ant_dict and ant_dict[tid]:
                        ant_rows = []
                        for vid in random.sample(ant_dict[tid], k=min(args.neighbors, len(ant_dict[tid]))):
                            mat_t = sense_dict.get(int(vid))
                            if mat_t is not None:
                                ant_rows.append(mat_t)
                        if tid in sense_dict:
                            ant_rows.append(sense_dict[tid])
                        if ant_rows:
                            ant_mat = torch.vstack(ant_rows).to(student_device)
                            if args.kappa < 99:
                                ant_mat = replace_sense_topk(teacher_target, ant_mat, "l2",
                                                            args.kappa, mode="near")[1]

              
                    tmp_losses.append(syn_ant_loss_no_norm(target, syn_mat, ant_mat, margin=args.hinge))

                # positions for tids
                valid = [(i, tid) for i, tid in enumerate(sense_ids)]
                pick = random.sample(valid, k=sample_length)
                #print("total_length", len(sense_ids), "len sense keys", len(valid))

                for pos, tid in pick:
                    target = stud_hidden[0, pos, :]
                    teacher_target = teach_hidden[ pos, :]

                    syn_mat = None
                    ant_mat = None
                    if  tid in sense_dict:
                        syn_mat = sense_dict[tid].to(student_device)
                        if args.kappa < 99:
                            syn_mat = replace_sense_topk(teacher_target, syn_mat, "l2",
                                                            args.kappa, mode="near")[1]
                
                    tmp_losses.append(syn_ant_loss_no_norm(target, syn_mat, ant_mat, margin=args.hinge))



                if tmp_losses:
                    total_syn_ant = torch.stack(tmp_losses).mean()

                # KD + next-token CE on combined_ids
                stud_logits = student_out.logits[0, :L_kd, :]
                #teach_logits = teach_logits.to(student_device)

                ce = loss_tok(stud_logits, sense_tokens[0,:].to(student_device))
                kd = kd_loss_logits(stud_logits, teach_logits)
                loss_main = kd + ce
                loss =  loss_main + args.sense_weight * total_syn_ant
                print(f"[train Loss0/1]  loss {loss.item():.4f} total_syn_ant {total_syn_ant.item():.4f} ")
 

            elif args.loss_type == 99: #  KD
                # KD Loss
                stud_logits = student_out.logits[0, :L_kd, :]
                #teach_logits = teach_logits.to(student_device)
       
                ce = loss_tok(stud_logits, sense_tokens[0,:].to(student_device))
                kd = kd_loss_logits(stud_logits, teach_logits)
                loss = kd + args.alpha * ce
                print(f"[train 99]  loss {loss.item():.4f}  ")
 

         
    
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Save at end of epoch (last few epochs in original; here: always ok)
        if rank == 0:
            ckpt = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'loss': loss.detach().cpu(),
            }
            torch.save(ckpt, f"{args.ckpt_path}{epoch}.pth")
            print(f"[train] saved checkpoint: {args.ckpt_path}{epoch}.pth")

        dist.barrier()
