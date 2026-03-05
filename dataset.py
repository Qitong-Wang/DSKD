from torch.utils.data import Dataset
import json
import random
from typing import List, Dict, Optional
from datasets import load_dataset

class MultiQADataset(Dataset):


    def __init__(
        self,
        args,
        dataset: str = "csqa",
        split: str = "train",
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.nshot = 5
        self.model_name = args.model_name


        # Load datasets (HF or JSON for MMLU)
        if self.dataset == "csqa":
            hf = load_dataset("tau/commonsense_qa")
        elif self.dataset == "piqa":
            hf = load_dataset("lighteval/piqa","plain_text")

        elif self.dataset in ("arc", "arc_challenge", "ai2_arc"):
            cfg =  "ARC-Challenge"
            hf = load_dataset("allenai/ai2_arc", cfg)
        elif self.dataset == "mmlu":
            if split == "train":
                with open("./dataset/MMLU/all_train.json", "r", encoding="utf-8") as f:
                    mmlu_list = json.load(f)
            else:
                with open("./dataset/MMLU/all_test.json", "r", encoding="utf-8") as f:
                    mmlu_list = json.load(f)
            # Emulate HF mapping with the provided split; also keep a 'train' pool for few-shot.
            hf = {split: mmlu_list, "train": mmlu_list}
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Normalize examples
        if split not in hf:
            raise ValueError(f"Split '{split}' not found. Available: {list(hf.keys())}")
        self.data: List[Dict] = [self._normalize_example(ex) for ex in hf[split]]

        if "train" not in hf:
            self.train_pool: List[Dict] = [self._normalize_example(ex) for ex in hf[split]]
        else:
            self.train_pool: List[Dict] = [self._normalize_example(ex) for ex in hf["train"]]

        # Build optional category index for category-aware few-shot (MMLU etc.)
        self._category_index: Dict[str, List[int]] = {}
        for i, ex in enumerate(self.train_pool):
            cat = ex.get("category")
            if cat:
                self._category_index.setdefault(cat, []).append(i)

        self.label_scheme = None 
        if self.dataset == "csqa":
            self.label_scheme = ["A", "B", "C", "D", "E"]
        if self.dataset == "piqa":
            self.label_scheme = ["0", "1"]
        if self.dataset in ("arc", "arc_challenge", "ai2_arc"):
            self.label_scheme = ["A", "B", "C", "D"]
        if self.dataset == "mmlu":
            self.label_scheme = ["A", "B", "C", "D"]

        self.label_token = None
        if self.model_name == "llama":
            if self.dataset == "csqa":
                self.label_token = [32,33,34,35,36]
            if self.dataset == "piqa":
                self.label_token = [15,16]
            if self.dataset in ("arc", "arc_challenge", "ai2_arc"):
                self.label_token = [32,33,34,35]
            if self.dataset == "mmlu":
                self.label_token = [32,33,34,35]
        if self.model_name == "mistral":
            if self.dataset == "csqa":
                self.label_token = [330,365,334,384,413]
            if self.dataset == "piqa":
                self.label_token = [28705,28734,28740]
            if self.dataset in ("arc", "arc_challenge", "ai2_arc"):
                self.label_token = [330,365,334,384]
            if self.dataset == "mmlu":
                self.label_token = [330,365,334,384]


        # Dataset-specific system prompt
        self.system_prompt = self._make_system_prompt()




    def _make_system_prompt(self) -> str:
        labels = ",".join(self.label_scheme)  # FIXED
   
        if self.dataset == "csqa":
            return (f"Answer the commonsense multiple-choice question. "
                    f"Reply with only one letter among: {labels}.")
        if self.dataset == "piqa":
                return (f"Answer the commonsense multiple-choice question. "
                    f"Reply with only one letter among: {labels}.")
        if self.dataset in ("arc", "arc_challenge", "ai2_arc"):
            return (f"Answer the grade-school science multiple-choice question. "
                    f"Reply with only one letter among: {labels}.")
        if self.dataset == "mmlu":
            return (f"Answer the multiple-choice question. "
                    f"Reply with only one letter among: {labels}.")
        return (f"Answer the multiple-choice question. "
                f"Reply with only one symbol among: {labels}.")



    # -------------------- Normalization per dataset --------------------
    def _normalize_example(self, ex: Dict) -> Dict:
        """
        Map raw example -> {id, question, options[List[str]], labels[List[str]], answer:str|None, category?:str}
        """
        if self.dataset == "csqa":
            raw_labels = list(ex["choices"]["label"])
            labels = sorted(raw_labels)  # ["A","B","C","D","E"]
            m = {L: T for L, T in zip(ex["choices"]["label"], ex["choices"]["text"])}
            options = [m[L] for L in labels]
            ans = ex.get("answerKey")
            ans = ans if (ans in labels) else None
            return {
                "id": ex.get("id", str(hash(ex["question"]))),
                "question": ex["question"],
                "options": options,
                "labels": labels,
                "answer": ans
            }

        if self.dataset == "piqa": 
            options = [ex["sol1"], ex["sol2"]]
            labels = ["0", "1"]   # keep exactly 0/1
            q = f"{ex['goal']}\nWhich solution works best?"
            lab = ex.get("label", None)  # HF PiQA provides int 0 or 1
            if isinstance(lab, str):
                try:
                    lab = int(lab)
                except Exception:
                    lab = None
            ans = str(lab) if lab in (0, 1) else None  # keep "0" or "1"
            ex_id = ex.get("id") or str(hash(ex['goal'] + ex['sol1'] + ex['sol2']))
            return {"id": ex_id, "question": q, "options": options,
                    "labels": labels, "answer": ans}

        if self.dataset in ("arc", "arc_challenge", "ai2_arc"):
            raw_labels = list(ex["choices"]["label"])
            labels = sorted(raw_labels)  # ["A","B","C","D"]
            m = {L: T for L, T in zip(ex["choices"]["label"], ex["choices"]["text"])}
            options = [m[L] for L in labels]
            gold = ex.get("answerKey")
            ans = gold if (gold in labels) else None
            q = ex["question"]
            ex_id = ex.get("id") or str(hash(q))
            return {"id": ex_id, "question": q, "options": options, "labels": labels, "answer": ans}

        if self.dataset == "mmlu":
            labels = ["A", "B", "C", "D"]
            options = list(ex["options"])
            if len(options) != 4:
                options = (options + [""] * 4)[:4]
            ans = ex.get("answer")
            ans = ans if ans in labels else None
            cat = ex.get("category")
            q = ex["question"]
            ex_id = ex.get("id") or str(hash(q + "".join(options)))
            return {
                "id": ex_id,
                "question": q,
                "options": options,
                "labels": labels,
                "answer": ans,
                "category": cat
            }

        raise ValueError(f"Unsupported dataset: {self.dataset}")

    # -------------------- Dataset plumbing --------------------
    def __len__(self):
        return len(self.data)

    @staticmethod
    def _format_prompt_block(question: str, options: List[str], labels: List[str]) -> str:
        text = question.rstrip() + "\n"
        for lab, opt in zip(labels, options):
            text += f"{lab}. {opt}\n"
        return text

    def _pick_exemplar_indices(self, row: Dict, k: int) -> List[int]:
        """
        Prefer same-category sampling when 'category' is present (e.g., MMLU).
        Falls back to global sampling otherwise. Avoids the same example id.
        """
        if k <= 0 or len(self.train_pool) <= 1:
            return []

        cat = row.get("category")
        if cat and cat in self._category_index:
            candidates = [i for i in self._category_index[cat] if self.train_pool[i]["id"] != row["id"]]
        else:
            candidates = [i for i in range(len(self.train_pool)) if self.train_pool[i]["id"] != row["id"]]

        if not candidates:
            return []
        k = min(k, len(candidates))
        return random.sample(candidates, k)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        prompt = [{"role": "system", "content": self.system_prompt}]

        # Few-shot exemplars
        if self.nshot > 0 and len(self.train_pool) > 0:
            ex_ids = self._pick_exemplar_indices(row, self.nshot)
            for ei in ex_ids:
                e = self.train_pool[ei]
                e_text = self._format_prompt_block(e["question"], e["options"], e["labels"])
                prompt.append({"role": "user", "content": e_text})
                prompt.append({"role": "assistant", "content": e.get("answer") or ""})

        # Target item
        tgt_text = self._format_prompt_block(row["question"], row["options"], row["labels"])
        prompt.append({"role": "user", "content": tgt_text})
       
        
        # Return gold label (may be None on test) + compatibility flag
        return prompt, row.get("answer"), self.dataset  , self.label_token

class MMLUTestDataset(Dataset):


    def __init__(self, args, test_json_path, dev_json_path=None):
        super().__init__()
        
        # Load test data
        with open(test_json_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        # Load dev data (optional)
        self.dev_data = None
        if dev_json_path:
            with open(dev_json_path, 'r', encoding='utf-8') as f:
                self.dev_data = json.load(f)

        self.nshot = 5

        # If no dev, build category dict from test for self-shot
        if self.dev_data is None:
            self.category_dict = {}
            for idx, row in enumerate(self.test_data):
                category = row['category']
                if category not in self.category_dict:
                    self.category_dict[category] = []
                self.category_dict[category].append(idx)


        self.model_name = args.model_name
        #self.label_token = [32,33,34,35]

        if self.model_name == "llama":
            self.label_token = [32,33,34,35]
        if self.model_name == "mistral":
            self.label_token = [330,365,334,384]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
       
        prompt_data = [{
            'role': 'system',
            'content': 'You are a helpful assistant that answers multiple-choice questions. Provide only the correct option letter (e.g., A, B, C, or D).'
        }]

        # Determine source for n-shot examples
        source_data = self.dev_data if self.dev_data else self.test_data
        if self.nshot > 0:
            if self.dev_data:
                indices = random.sample(range(len(self.dev_data)), min(len(self.dev_data), self.nshot))
            else:
                category = self.test_data[idx]['category']
                indices = random.sample(self.category_dict[category], min(len(self.category_dict[category]), self.nshot))
            
            for i in indices:
                row = source_data[i]
                q = row['question']
                opts = row['options']
                ans = row['answer']
                text = f"{q}\n" + "\n".join(f"{label}. {opt}" for label, opt in zip(['A', 'B', 'C', 'D'], opts))
                prompt_data.append({'role': 'user', 'content': text})
                prompt_data.append({'role': 'assistant', 'content': ans})

        # Add target question
        row = self.test_data[idx]
        q = row['question']
        opts = row['options']
        ans = row['answer']
        text = f"{q}\n" + "\n".join(f"{label}. {opt}" for label, opt in zip(['A', 'B', 'C', 'D'], opts))
        prompt_data.append({'role': 'user', 'content': text})

        return prompt_data, ans,   "mmlu"  , 1




class SquadDataset(Dataset):
    def __init__(self,args, json_path ) :
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.model_name = args.model_name
        self.samples = []
        for article in data['data']:
            for para in article['paragraphs']:
                context = para['context']
                qas = para['qas']
                # Build question-answer pairs for this paragraph
                qa_pairs = []
                for qa in qas:
                    answers = [a['text'] for a in qa['answers']]
                    qa_pairs.append({
                        'question': qa['question'],
                        'answers': answers,
                        'id': qa['id'],
                        'context': context,
                        'is_impossible': qa['is_impossible']
                    })
                self.samples.extend(qa_pairs)
        
        self.nshot = 1
       

        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        context = sample['context']
        current_q = sample['question']
        current_answers = sample['answers']  # list of texts

        # Sample one prior QA for 1-shot (same context, different ID, has answers)
        candidates = [s for s in self.samples if s['context'] == context and s['id'] != sample['id'] and s['answers']]
        if candidates:
            one_shot_sample = random.choice(candidates)
           
            one_shot_q = one_shot_sample['question']
            one_shot_a = one_shot_sample['answers'][0]  # use first as 1-shot example
            one_shot_impossible = one_shot_sample['is_impossible']
        else:
            one_shot_q = None
            one_shot_a = None

        prompt_data = []
        if self.model_name == "llama" :
            prompt_data.append({
                'role': 'system',
                'content': 'You are a helpful assistant. Answer the question based on the given context. Provide a short, precise answer. If the answer is not present in the context, respond exactly with: No answer '
            })
        elif self.model_name in [ "granite"]:
            

            prompt_data.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user question strictly using the provided context. "
                    "Always respond with either:\n"
                    "1. The exact answer span from the context, with no added words.\n"
                    "2. The text: No answer"
                )
            })
        elif self.model_name == "mistral":

            prompt_data.append({
                "role": "system",
                "content": (
                    "You are a helpful assistant for extractive QA.\n"
                    "Given the CONTEXT and QUESTION, output exactly one of the following:\n"
                    "the exact answer span copied verbatim from the context.\n"
                    "two words: No answer\n"
                    "\n"
                    "Do not output anything else. Do not explain."
                )
            })
                       

         

        if self.nshot == 1:
            if one_shot_q:
                prompt_data.append({
                    'role': 'user',
                    'content': f"Context: {context}\n\nQuestion: {one_shot_q}\n"
                })
                if one_shot_impossible:
                  
                    one_shot_a = "No answer"


                prompt_data.append({
                    'role': 'assistant',
                    'content': f"{one_shot_a}"
                })
                prompt_data.append({
                    'role': 'user',
                    'content': f"Question: {current_q}"
                })
            else:
                prompt_data.append({
                    'role': 'user',
                    'content': f"Context: {context}\n\nQuestion: {current_q}"
                })
        elif self.nshot == 0:
            prompt_data.append({
                'role': 'user',
                'content': f"Context: {context}\n\nQuestion: {current_q}"
            })
        else:
            raise ValueError("nshot should be 0 or 1 for SQuAD dataset.")

        
        return prompt_data, current_answers, "squad", sample['is_impossible']




class WikiDataset(Dataset):
    """
    Each row is assumed to be:
        [question, optionA, optionB, optionC, optionD, correct_answer_letter]
    """

    def __init__(self,  args,json_path):
        super().__init__()

        with open(json_path, 'r', encoding='utf-8') as f:
            self.train_data = json.load(f)
        self.data = self.train_data
  

       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt_data = self.data[idx]
        prompt_data = ''.join(prompt_data)

        return prompt_data , [] , "wiki", 1
