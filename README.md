# Sense Dictionary Knowledge Distillation (DSKD)

This repository contains the **training and evaluation scripts** used in our paper on **Sense Dictionary Knowledge Distillation (DSKD)** for large language models.

The purpose of this README is to **enable reproducibility of the main experimental results** reported in the paper.  
This README focuses **only on model training and evaluation**.

> **Note**  
> If you are interested in how datasets and sense dictionaries are constructed, please refer to **`README_prep.md`**.  
> That preprocessing pipeline is **not required** if you only want to reproduce training and evaluation results.

---

## Overview

The core experiments are driven by **three shell scripts**:

| Script | Purpose |
|------|--------|
| `run_train_llama.sh` | Train and evaluate a DSKD student model with a **LLaMA** backbone |
| `run_train_mistral.sh` | Train and evaluate a DSKD student model with a **Mistral** backbone |
| `run_inference.sh` | Run inference-only evaluation using trained checkpoints |

All scripts use **`torchrun`** for multi-GPU distributed execution.

---

## Data Preprocessing

The link to dataset: https://drive.google.com/file/d/1KcEj6uYaZK0r1dQFUxCcCUbitQiCRTew/view?usp=sharing 

The link to Llama sense dictionary: https://drive.google.com/file/d/1Sa6L3JMY8y-D08AHcHOo-eGpomi7HPks/view?usp=sharing

The link to Mistral sense dictionary: https://drive.google.com/file/d/1a6X1qFCskx6cAAnOTy77Y_elSDXXT2VZ/view?usp=sharing

The link to Llama checkpoint: https://drive.google.com/file/d/12XWntyTQrDo2yHyWbNU-oUX201iSQQ7m/view?usp=sharing 

The link to Mistral checkpoint: https://drive.google.com/file/d/1pu7U9Zv3kaBJW3AT-KUIqSvuDbuBhgnm/view?usp=sharing

If you want to reproduce:
- Sense dictionary construction
- Synonym / antonym extraction
- Token- and span-level sense alignment

Please refer to:

```text
README_prep.md
```


---

## Training

### Train with LLaMA Backbone

```bash
bash run_train_llama.sh
```

This script:
- Trains a **16-layer student LLaMA model** using DSKD
- Uses SQuAD v2 as the training dataset
- Incorporates sense dictionaries, synonym and antonym relations
- Saves checkpoints to `./ckpt/`
- Evaluates the trained model on:
  - ARC
  - CSQA
  - PIQA
  - MMLU
  - SQuAD

---

### Train with Mistral Backbone

```bash
bash run_train_mistral.sh
```

This script:
- Trains a **16-layer student Mistral model** under the same DSKD framework
- Uses backbone-specific sense dictionaries and lexical resources
- Runs the same evaluation suite as the LLaMA setup

---

## Inference

```bash
bash run_inference.sh
```

This script:
- Runs inference on:
  - ARC
  - CSQA
  - PIQA
  - MMLU
  - SQuAD
- Evaluates **both LLaMA and Mistral student models** by provided checkpoints


Expected checkpoints:


