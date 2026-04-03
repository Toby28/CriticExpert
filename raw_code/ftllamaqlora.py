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
from dataset import mamual, medical_dataset, mental_dataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer, AutoConfig, LlamaModel, PreTrainedModel,BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import tqdm
import pandas as pd
import numpy as np
from peft import prepare_model_for_kbit_training


# --------------------------- 1. CLI arguments --------------------------------

# --------------------------- 2. Dataset helpers ------------------------------
'''
class medical_dataset:
    def __init__(self):
        self.label_map = self.get_label()
        self.train, self.train_label = self.get_trainset()
        self.test, self.test_label = self.get_testset()
        self.definition = [
            "Neoplasms are abnormal growths of tissue that result from uncontrolled, excessive cell division and can be benign or malignant (cancerous).",
            "Digestive system diseases are disorders that affect the organs responsible for breaking down food, absorbing nutrients, and eliminating waste from the body.",
            "Nervous system diseases are disorders that affect the brain, spinal cord, or nerves, disrupting normal neurological function.",
            "Cardiovascular diseases are a group of disorders affecting the heart and blood vessels, including conditions such as heart attacks, strokes, and hypertension.",
            "General pathological conditions refer to common disease-related changes or abnormalities in tissues and organs that affect the body as a whole, regardless of specific diseases.",
        ]
        self.index2label = ["Neoplasms", "Digestive system diseases",
                            "Nervous system diseases", "Cardiovascular diseases",
                            "General pathological conditions"]
        self.label2index = {"Neoplasms": 0, "Digestive system diseases": 1,
                            "Nervous system diseases": 2, "Cardiovascular diseases": 3,
                            "General pathological conditions": 4}

    def get_label(self):
        df = pd.read_csv('./dataset2/medical/medical_tc_labels.csv')
        label_map = {}
        for i in range(len(df)):
            label_map[df.iloc[i, 0]] = df.iloc[i, 1]
        return label_map

    def get_trainset(self):
        train_list = []
        train_label = []
        df = pd.read_csv('./dataset2/medical/medical_tc_train.csv')
        print(df.head())
        for i in range(len(df)):
            train_list.append(df.iloc[i, 1])
            train_label.append(df.iloc[i, 0])
        print(len(train_list))
        return train_list, train_label

    def get_testset(self):
        df = pd.read_csv('./dataset2/medical/medical_tc_test.csv')
        print(df.head())
        return df.iloc[:, 1], df.iloc[:, 0]

    def get_llama_results(self):
        pass

    def get_chatgpt4o_results(self):
        train_ds = self.load_jsonl('./clean/med_gpt_v6_train_clean.json', flag="train")
        val_ds = self.load_jsonl('./clean/med_gpt_v6_test_clean.json', flag='test')

        return train_ds, val_ds

    def load_jsonl(self, path, flag=None):
        if flag == "train":
            data = self.train
            data_label = self.train_label
            with open(path, 'r') as file:
                dialogue = json.load(file)
        elif flag == "test":
            data = self.test
            data_label = self.test_label
            with open(path, 'r') as file:
                dialogue = json.load(file)

        rows = []
        for i in range(len(dialogue['MED_Code'])):
            # for i in range(500):
            if dialogue["MED_Code"][i] is not None and dialogue['MED_Code'][i] in self.label2index.keys():
                tp_prompt = ""
                tp_prompt += f"Dialogue: {data[i]},"
                tp_prompt += f"AI-Code: {dialogue['MED_Code'][i]},"
                tp_prompt += f"Explanation: {dialogue['Explanation'][i]},"
                tp_prompt += f"Definition: {self.definition[self.label2index[dialogue['MED_Code'][i]]]}"

                if self.label2index[dialogue['MED_Code'][i]] == data_label[i] - 1:
                    rows.append({"text": tp_prompt,
                                 "labels": 1,
                                 "labels2": data_label[i] - 1
                                 })
                else:
                    rows.append({"text": tp_prompt,
                                 "labels": 0,
                                 "labels2": data_label[i] - 1
                                 })
        return Dataset.from_list(rows)

'''
def tok_fn(batch):
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.pad_token = tok.eos_token  # Llama has no dedicated PAD token

    return tok(batch["text"],
               truncation=True, padding="max_length",
               max_length=args.max_length)


# --------------------------- 4. Custom model ---------------------------------
class Llama3ForCustomCLS(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, base_model, num_labels=2,
                 lora_r=128, lora_alpha=32,
                 target_modules=("q_proj", "k_proj", "v_proj")):
        super().__init__(base_model.config)

        # ❷ LoRA adapters

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=0.05, bias="none",
            target_modules=list(target_modules),
        )

        self.backbone = base_model
        # self.backbone = get_peft_model(self.backbone, lora_cfg)
        # ❸ Classification head
        hidden = base_model.config.hidden_size  # 4 096 for Llama-3-8B
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden, hidden // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden // 2, num_labels),
        # )

        self.pre_classifier1 = torch.nn.Linear(hidden // 2, hidden // 2)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier1 = torch.nn.Linear(hidden //2, 2)
        self.pre_classifier2 = torch.nn.Linear(hidden, hidden // 2)
        self.classifier2 = torch.nn.Linear(hidden//2, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, labels2=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = out.last_hidden_state[:, 5]  # CLS-token pooling
        pooler2 = self.pre_classifier2(pooled)
        pooler2 = torch.nn.ReLU()(pooler2)
        # pooler2 = torch.nn.GELU()(pooler2)
        pooler2 = self.dropout(pooler2)
        output2 = self.classifier2(pooler2)
        #
        pooler1 = self.pre_classifier1(pooler2)
        pooler1 = torch.nn.ReLU()(pooler1)
        # pooler1 = torch.nn.GELU()(pooler1)
        pooler1 = self.dropout(pooler1)
        output1 = self.classifier1(pooler1)

        # logits = self.classifier(pooled)
        # if labels is not None:

        # loss = self.loss_fn(output1, labels) + 0.1 * self.loss_fn(output2, labels2)
        loss1 = self.loss_fn(output1, labels)
        loss2 = self.loss_fn(output2, labels2)
        return {"loss": loss1 + loss2, "logits1": output1, "logits2": output2}
        # return {"loss": loss, "logits1": output1}
        # return {"logits": logits}


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
    args.add_argument("--train_file", default="data/train.txt")
    args.add_argument("--val_file", default="data/val.txt")
    # args.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B")
    # args.add_argument("--model_name", default="meta-llama/Llama-3.2-3B-Instruct")
    # args.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    args.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")

    args.add_argument("--num_labels", type=int, default=10,
                      help="If omitted, infer from data")
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--grad_accum", type=int, default=1)
    args.add_argument("--lr", type=float, default=2e-3)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--output_dir", default="llama3-8b-textcls")
    args.add_argument("--device", default="cuda:1",
                      help="GPU index, e.g. '0' or '0,1'. Set before torch import.")
    args = args.parse_args()
    # medical = medical_dataset()
    medical = mamual()
    torch.cuda.set_device(args.device)
    # --------------------------- 5. Training loop --------------------------------
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )

    base = LlamaModel.from_pretrained(
        args.model_name,
        # quantization_config=bnb_cfg,
        device_map=args.device
    )

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    model = Llama3ForCustomCLS(base, args.num_labels)
    params = list(model.pre_classifier1.parameters()) + list(model.classifier1.parameters()) + list(model.pre_classifier2.parameters()) + list(model.classifier2.parameters())

    # optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim = torch.optim.AdamW(params, lr=args.lr)

    # train_ds = load_jsonl(args.train_file)
    # val_ds = load_jsonl(args.val_file)

    train_ds, val_ds = medical.get_llama_results()
    print(train_ds, val_ds)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size)

    model, optim, train_dl, val_dl = accelerator.prepare(
        model, optim, train_dl, val_dl)

    # --------------------------- 3. Tokeniser ------------------------------------

    res1 = None
    res2 = None
    tar1 = None
    tar2 = None
    max_acc = 0.0
    for epoch in range(args.epochs):
        # ----- train -----
        model.train()
        total_loss, step = 0, 0
        correct, seen = 0, 0
        for batch in tqdm.tqdm(train_dl):
            optim.zero_grad()
            # print(batch)
            tensor_batch = {k: v for k, v in batch.items()
                            if k in ("input_ids", "attention_mask", "labels", "labels2")}

            out = model(**tensor_batch)
            accelerator.backward(out["loss"] / args.grad_accum)
            if (step + 1) % args.grad_accum == 0:
                optim.step()
            total_loss += out["loss"].item()
            step += 1

            logits1 = out["logits1"]
            # print(logits1)
            preds1 = logits1.argmax(-1)
            # print(preds1)
            # print(tensor_batch["labels"])
            correct += (preds1 == tensor_batch["labels"]).sum().item()
            seen += len(preds1)

        accelerator.print(f"[epoch {epoch}] train loss "
                          f"{total_loss / step:.4f}")
        accelerator.print(f"[epoch {epoch}] train acc "
                          f"{(correct / seen):.3%}")
        # ----- validation -----
        model.eval()
        correct, seen = 0, 0
        res_pre1, res_tar1 = [], []
        res_pre2, res_tar2 = [], []
        with torch.no_grad():
            for batch in val_dl:
                tensor_batch = {k: v for k, v in batch.items()
                                if k in ("input_ids", "attention_mask", "labels", "labels2")}
                tp = model(**tensor_batch)
                logits1 = tp["logits1"]
                logits2 = tp["logits2"]
                preds1 = logits1.argmax(-1)
                preds2 = logits2.argmax(-1)
                # print(preds1)
                # print(tensor_batch["labels"])
                for kk in range(len(logits1)):
                    res_pre1.append(preds1[kk].item())
                    if tar1 == None:
                        res_tar1.append(tensor_batch["labels"][kk].item())

                    res_pre2.append(preds2[kk].item())
                    if tar2 == None:
                        res_tar2.append(tensor_batch["labels2"][kk].item())

                correct += (preds1 == tensor_batch["labels"]).sum().item()
                seen += len(preds1)

            if tar1 == None:
                tar1 = res_tar1

            if tar2 == None:
                tar = res_tar2

            if (max_acc <  (correct / seen)):
                    max_acc=correct / seen
                    res1 = res_pre1
                    res2 = res_pre2
        accelerator.print(f"[epoch {epoch}] val acc "
                          f"{(correct / seen):.3%}")

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
    res = {
        "pred1": res1,
        "targ1": tar1,
        "pred2": res2,
        "targ2": tar2,
    }

    np.save("mannul_llama_llmQlora", res)
    # Set CUDA device *before* Torch / Bits-and-Bytes load anything
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

