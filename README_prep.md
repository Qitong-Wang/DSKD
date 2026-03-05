# DSKD Word Relationship Preparation Pipeline

This README provides instructions for the **word relationship preparation pipeline** used in **Sense Dictionary Knowledge Distillation (DSKD)**.
This pipeline is required only for full reproducibility of the sense dictionary construction process. If you are interested in training or evaluating the student models, you may safely skip this pipeline and directly use the provided files released with this repository.

---

## Overview

**Input sources**
- Dict2Vec word-pair lists (TXT)
- Wiktionary dump (Kaikki / Wiktextract JSONL)
- MorphoLEX morphology database (XLSX)


---

## Pipeline Order (Recommended)

```
prep0 → prep1 -> prep2 → prep3 
```
The numeric prefix in each filename (e.g., prep0, prep1, prep2) indicates the execution order.

---

## Step 0A 

**Purpose**  
Convert a Dict2Vec-style word-pair TXT file into a bidirectional JSON dictionary.

**Example**
```bash
python prep0_generate_dict2vec_pairs.py   --input_txt ./data_source/ant-pairs.txt   --output_json ./relationship/dict2vec_ant.json
```

---

## Step 0B

**Purpose**  
Extract English synonym and antonym relations from a Wiktionary JSONL dump.

**Example**
```bash
python prep0_generate_wiktionary_pairs.py   --jsonl ./data_source/kaikki.org-dictionary-English.jsonl   --out-dir ./relationship
```

---

## Step 1

**Purpose**  
Merge two relation dictionaries into one unified map.

**Example**
```bash
python prep1_combine_pairs.py   --json1 ./relationship/english_ant.json   --json2 ./relationship/dict2vec_ant.json   --output_json ./relationship/ant.json
```

---

## Step 2 

**Purpose**  
Normalize morphological negation and flip polarity when needed.

**Example**
```bash
python prep2_addjsonpolar.py   --morpholex_xlsx ./data_source/MorphoLEX_en.xlsx   --in_syn_json ./relationship/syn.json   --in_ant_json ./relationship/ant.json
```

---

## Step 3 


**Purpose**  
Decompose-word Export

**Mistral**
```bash
python prep3_decompose_word.py   --model mistralai/Mistral-7B-Instruct-v0.1   --mmax 3
```

**LLaMA**
```bash
python prep3_decompose_word.py   --model meta-llama/Meta-Llama-3-8B-Instruct   --mmax 3
```

---

## Notes
- Run Step 3 separately for each tokenizer.
- Outputs are intended for downstream training or inference.

## Step 4

**Purpose**  
Build Sense Dictionary

**Mistral**
```bash
sh build_sense_mistral.sh
```
**LLaMA**
```bash
sh build_sense_llama.sh
```