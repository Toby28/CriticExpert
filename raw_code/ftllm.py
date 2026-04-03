import torch, torch.nn as nn
from transformers import LlamaModel, AutoConfig, PreTrainedModel
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import torch, torch.nn as nn
from transformers import LlamaModel, AutoConfig, PreTrainedModel
from peft import LoraConfig, get_peft_model

import torch, os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # optional, but keeps things tidy
from huggingface_hub import login

from transformers import pipeline
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# !/usr/bin/env python
# train_llama3_textcls.py
"""
Fine-tune Meta-Llama-3-8B on a text-classification task with QLoRA + custom head.
Requires: transformers>=4.46.3, peft>=0.13.2, bitsandbytes, accelerate, datasets, flash-attn, torch>=2.4
"""

import os, json, argparse, pathlib, torch
from dataset2 import mamual, pcit, medical_dataset, mental_dataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoConfig, LlamaModel, PreTrainedModel,BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,TaskType
from trl import SFTTrainer
from datasets import Dataset
from peft import prepare_model_for_kbit_training


# --------------------------- 1. CLI arguments --------------------------------

# --------------------------- 2. Dataset helpers ------------------------------
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.samples = data
        self.tok = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
        self.tok.pad_token = self.tok.eos_token
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        full_prompts = sample['gt']
        prompts_without_response = sample['text']

        tokenized_full_without_prompts = self.tok(prompts_without_response,
                                             truncation=True, padding="max_length",
                                             max_length=args.max_length, return_tensors="pt")
        # print(batch['text'], batch['gt'])
        tokenized_full_prompts = self.tok(full_prompts,
                                     truncation=True, padding="max_length",
                                     max_length=args.max_length, return_tensors="pt")

        labels = tokenized_full_prompts["input_ids"].clone()
        prompt_lengths = tokenized_full_without_prompts["attention_mask"].sum(dim=1)

        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100


        input_ids = tokenized_full_prompts['input_ids'].squeeze(0)
        attention_mask = tokenized_full_prompts['attention_mask'].squeeze(0)
        labels = labels.squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



def tok_fn(batch, mp):
    tok = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tok.pad_token = tok.eos_token  # Llama has no dedicated PAD token
    print(len(batch))
    full_prompts = [
        f"{batch['gt'][i]}" for i in range(len(batch))
    ]

    # 2. Format the prompt without the response (for finding the split point)
    prompts_without_response = [
        f"{batch['text'][i]}" for i in range(len(batch))
    ]
    # print(full_prompts)
    tokenized_full_without_prompts = tok(prompts_without_response,
               truncation=True, padding="max_length",
               max_length=args.max_length, return_tensors="pt")
    # print(batch['text'], batch['gt'])
    tokenized_full_prompts = tok(full_prompts,
               truncation=True, padding="max_length",
               max_length=args.max_length,return_tensors="pt")

    labels = tokenized_full_prompts["input_ids"].clone()
    prompt_lengths = tokenized_full_without_prompts["attention_mask"].sum(dim=1)

    for i in range(len(labels)):
        labels[i, :prompt_lengths[i]] = -100

    tokenized_full_prompts['labels'] = labels
    return tokenized_full_prompts
# --------------------------- 4. Custom model ---------------------------------



# --------------------------- 6. Save artefacts -------------------------------
# save_dir = pathlib.Path(args.output_dir)
# save_dir.mkdir(exist_ok=True)

# (a) LoRA adapters + backbone config
# model.backbone.save_pretrained(save_dir / "lora_backbone")
# (b) Classification head
# torch.save(model.classifier.state_dict(),
#            save_dir / "cls_head.pt")
# (c) Tokeniser
# tok.save_pretrained(save_dir / "lora_backbone")

# accelerator.print(f"Saved LoRA adapters + head to {save_dir}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", default="dpics")
    args.add_argument("--method", default="mix")
    # args.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    # args.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    # args.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")

    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--grad_accum", type=int, default=1)
    args.add_argument("--lr", type=float, default=2e-4)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--output_dir", default="llama3-8b-textcls")
    args.add_argument("--device", default="cuda:0",
                      help="GPU index, e.g. '0' or '0,1'. Set before torch import.")
    args = args.parse_args()
    medical = None
    mp = None
    if args.dataset == "dpics":
        medical = mamual()
        mp = medical.index2label
    elif args.dataset == "pcit":
        medical = pcit()
        mp = medical.index2label
    elif args.dataset == "med":
        medical = medical_dataset()
        mp = medical.index2label
    elif args.dataset == "mental":
        medical = mental_dataset()
        mp = medical.label_map


    torch.cuda.set_device(args.device)

    train_ds, val_ds = None , None
    if args.method == "llama":
        train_ds, val_ds = medical.get_llama_results()
    elif args.method == "mix":
        train_ds, val_ds = medical.get_mixtral_results()
    elif args.method == "gpt":
        train_ds, val_ds = medical.get_chatgpt4o_results()

    print(train_ds, val_ds)
    # print(train_ds.shape)
    tar1= val_ds["labels1"]
    tar2 = val_ds["labels2"]
    train_ds = CustomDataset(train_ds)
    # train_ds = train_ds.map(tok_fn, mp, batched=True, batch_size=4, remove_columns=train_ds.column_names)
    # val_ds = val_ds.map(tok_fn, mp, batched=True, remove_columns=val_ds.column_names)
    # train_ds.set_format(type="torch")
    # val_ds.set_format(type="torch")

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size)
    # print(train_ds)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        # device_map={'':torch.cuda.current_device()}
        device_map =args.device
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=args.lr)
    

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.train()
    for epoch in tqdm.tqdm(range(args.epochs)):  # number of epochs

        for step, batch in enumerate(train_dl):
            batch = {k: v.to(args.device) for k, v in batch.items()
                            if k in ("input_ids", "attention_mask", "labels")}
            # print(batch.shape)
            outputs = model(**batch)
            loss = outputs.loss
    
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
            # if step % 100 == 0:
        print(f"Epoch {epoch} | Loss {loss.item()}")

    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

    # Create the prompt (must match the training format)
    # prompt = "### Instruction:\nRephrase the following sentence.\n\n### Input:\nHe quickly finished his assignment.\n\n### Response:\n"
    res_pre1 = []
    # Run inference
    for i in range(len(val_ds)):
        prompt = val_ds['text'][i]
        result = generator(prompt)
        res_pre1.append(result[0]['generated_text'])
    '''
    # ----- validation -----
    model.eval()
    correct, seen = 0, 0
    res_pre1 = []
    with torch.no_grad():
        for batch in val_dl:
            tensor_batch = {k: v.to(args.device) for k, v in batch.items()
                            if k in ("input_ids", "attention_mask", "labels")}
            tp = model.generate(**tensor_batch,
                                max_new_tokens=100,
                                temperature=0.2,
                                top_p=0.9,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                                )
            for i in range(len(tp)):
                generated_text = tokenizer.decode(tp[i], skip_special_tokens=True)
                print(generated_text)
            # print(tensor_batch["labels"])
            # for kk in range(len(generated_text)):
                res_pre1.append(generated_text)
                # if tar1 == None:
                #     res_tar1.append(tensor_batch["labels1"][kk].item())
                #     res_tar2.append(tensor_batch["labels2"][kk].item())

        # if tar1 == None:
        #     tar1 = res_tar1
        #     tar = res_tar2
    '''
    res = {
        "pred1": res_pre1,
        "tar1": tar1,
        "tar2":tar2
    }
    target_file = f"{args.dataset}_{args.method}_Qlora{args.epochs}_lr{args.lr}.json"
    with open(target_file, 'w') as f:
        json.dump(res, f)
    '''
    # ----- validation -----
    # model.eval()
    # correct, seen = 0, 0
    # res_pre1, res_tar1 = [], []
    # res_pre2, res_tar2 = [], []
    # with torch.no_grad():
    #     for batch in val_dl:
    #         tensor_batch = {k: v for k, v in batch.items()
    #                         if k in ("input_ids", "attention_mask", "labels", "labels2")}
    #
    #         tp = model(**tensor_batch)
    #         logits1 = tp["logits1"]
    #         logits2 = tp["logits2"]
    #         preds1 = logits1.argmax(-1)
    #         preds2 = logits2.argmax(-1)
    #         correct += (preds1 == tensor_batch["labels"]).sum().item()
    #         seen += len(preds1)
    #         for kk in range(len(logits1)):
    #             res_pre1.append(preds1[kk].item())
    #             res_tar1.append(tensor_batch["labels"][kk].item())
    #
    #             res_pre2.append(preds2[kk].item())
    #             res_tar2.append(tensor_batch["labels2"][kk].item())
    # accelerator.print(f"[epoch {epoch}] val acc "
    #                   f"{(correct / seen):.3%}")

    # Set CUDA device *before* Torch / Bits-and-Bytes load anything
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    '''
