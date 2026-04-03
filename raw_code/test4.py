import os, json, argparse, pathlib, torch
from dataset import mamual, pcit, medical_dataset, mental_dataset
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", default="mental")
    args.add_argument("--method", default="llama")

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


    train_ds, val_ds = None , None
    if args.method == "llama":
        val_ds = medical.get_llama_test_results()
    elif args.method == "mix":
        val_ds = medical.get_mixtral_test_results()
    elif args.method == "gpt":
        val_ds = medical.get_chatgpt4o_test_results()

    pred = []
    targ = []
    for i in range(len(val_ds)):
        pred.append(val_ds[i]['text'])
        targ.append(val_ds[i]['labels2'])

    print(accuracy_score(pred, targ))
    print(f1_score(targ, pred, average="macro"))