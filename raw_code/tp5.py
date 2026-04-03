import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
import torch
# import seaborn as sns
# import transformers
import json

from torch.nn.functional import dropout
from tqdm import tqdm
import argparse
import datetime
import logging
import sys
import logging
logging.basicConfig(level=logging.ERROR)
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
# from accelerate import init_empty_weights
import pickle
# Setting up the device for GPU usage
from torch import cuda
from torch.utils.data import Dataset, DataLoader
import copy
# from models.encoder.roberta import RobertlargeDataset, RobertalargeClass
from models.roberta import RobertlargeDataset, RobertalargeClass, RobertDataset, RobertaClass
# from models.encoder.roberta import RobertDataset, RobertaClass
from dataset import medical_dataset, mental_dataset, mamual, pcit
import time

def calcuate_accuracy(preds, targets,flag=0):
    n_correct = (preds==targets).sum().item()
    # if flag==1:
    #     print(preds)
    #     print(targets)
    #     print(preds==targets)
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model



def train(model,epoch,training_loader,loss_function,optimizer, device):
    tr_loss = 0
    n_correct1 = 0
    n_correct2 = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    # for _, data in tqdm(enumerate(training_loader, 0)):
    for data in tqdm(training_loader):
        print(data)
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        # if data['token_type_ids']:
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        output1,output2 = model(ids, mask, token_type_ids)
        # else:
        #     outputs = model(ids, mask)
        targets1 = data['targets1'].to(device, dtype=torch.long)
        targets2 = data['targets2'].to(device, dtype=torch.long)
        loss = loss_function(output1, targets1) + loss_function(output2, targets2)
        # loss = loss_function(output1, targets1)
        
        tr_loss += loss.item()
        big_val1, big_idx1 = torch.max(output1.data, dim=1)
        n_correct1 += calcuate_accuracy(big_idx1, targets1)
        
        big_val2, big_idx2 = torch.max(output2.data, dim=1)
        n_correct2 += calcuate_accuracy(big_idx2, targets2)

        nb_tr_steps += 1
        nb_tr_examples += targets1.size(0)

        # if _ % 5000 == 0:
        #     loss_step = tr_loss / nb_tr_steps
        #     accu_step = (n_correct1 * 100) / nb_tr_examples
        #     accu_step = (n_correct2 * 100) / nb_tr_examples
            #print(f"Training Loss per 5000 steps: {loss_step}")
            #print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct1 * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu1 = (n_correct1 * 100) / nb_tr_examples
    epoch_accu2 = (n_correct2 * 100) / nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu1}")
    print(f"Training Accuracy Epoch: {epoch_accu2}")

    return


def valid(model, testing_loader,flag=0,loss_function=torch.nn.CrossEntropyLoss(), device=None):
    model.eval()
    n_correct1 = 0
    n_correct2 = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    res_pre1, res_tar1 = [], []
    res_pre2, res_tar2 = [], []
    with torch.no_grad():
        # for _, data in tqdm(enumerate(testing_loader, 0)):
        for data in tqdm(testing_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            # if data['token_type_ids']:
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs1, outputs2 = model(ids, mask, token_type_ids)
            outputs1=outputs1.squeeze()
            outputs2=outputs2.squeeze()
            #else:
                 #outputs = model(ids, mask).squeeze()

            # print(outputs)
            # print(targets)
            targets1 = data['targets1'].to(device, dtype=torch.long)
            targets2 = data['targets2'].to(device, dtype=torch.long)

            loss = loss_function(outputs1, targets1) + loss_function(outputs2, targets2)
            # loss = loss_function(outputs1, targets1)
            tr_loss += loss.item()
            big_val1, big_idx1 = torch.max(outputs1.data, dim=1)
            n_correct1 += calcuate_accuracy(big_idx1, targets1,flag)

            big_val2, big_idx2 = torch.max(outputs2.data, dim=1)
            n_correct2 += calcuate_accuracy(big_idx2, targets2,flag)


            for kk in range(len(targets1)):
                res_pre1.append(big_idx1[kk].item())
                res_tar1.append(targets1[kk].item())
                
                res_pre2.append(big_idx2[kk].item())
                res_tar2.append(targets2[kk].item())

            nb_tr_steps += 1
            nb_tr_examples += targets1.size(0)

            '''
            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
            '''
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu1 = (n_correct1 * 100) / nb_tr_examples
    epoch_accu2 = (n_correct2 * 100) / nb_tr_examples

    if flag==2:
        print(f"testing Loss Epoch: {epoch_loss}")
        print(f"testing Accuracy Epoch: {epoch_accu1}")
        print(f"testing Accuracy Epoch: {epoch_accu2}")
    else:
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu1}")
        print(f"Validation Accuracy Epoch: {epoch_accu2}")

    return epoch_accu1, res_pre1, res_tar1, res_pre2, res_tar2


def main(args):

    # data_dir, dataset, method, device, SEED
    SEED = args.seed
    # random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # data_dir = args.dataset_path
    dataset = args.dataset
    batch_size = args.batchsize
    device = args.device
    llm = args.llm
    learning_rate = args.lr
    # weight_decay = args.weight_decay
    epoches = args.epoches
    method_name = args.method
    MAX_LEN = args.maxlen
    dropout = args.dropout
    optim=args.optim
    flag=args.flag
    outclass=10
    # cache_dir = args.cache_dir
    # load dataset
    train = None
    val = None
    if dataset=="mamual":
        medical = mamual()
        if llm=="llama":
            train, val = medical.get_llama_results()
        elif llm=="mixtral":
            train, val = medical.get_mixtral_results()
        elif llm=="chatgpt":
            train, val = medical.get_chatgpt4o_results()
        outclass = 10
    elif dataset=="pcit":
        medical = pcit()
        if llm=="llama":
            train, val = medical.get_llama_results()
        elif llm=="mixtral":
            train, val = medical.get_mixtral_results()
        elif llm=="chatgpt":
            train, val = medical.get_chatgpt4o_results()
        outclass = 8
    elif dataset=="medical":
        medical = medical_dataset()
        outclass = 5
        if llm=="llama":
            train, val = medical.get_llama_results()
        elif llm=="mixtral":
            train, val = medical.get_mixtral_results()
        elif llm=="chatgpt":
            train, val = medical.get_chatgpt4o_results()
    elif dataset=="mental":
        mental = mental_dataset()
        outclass = 7
        if llm=="llama":
            train, val= mental.get_llama_results()
        elif llm=="mixtral":
            train, val = mental.get_mixtral_results()
        elif llm=="chatgpt":
            train, val = mental.get_chatgpt4o_results()

    # new_df = full_data_preprocess("res/mamual_gpt.json")
    # new_df=clips_test_data_preprocess("raw2_new.pkl")

    # train_size = 1.0
    # train_data = new_df.sample(frac=train_size, random_state=2025)
    # # test_data = new_df.drop(train_data.index).reset_index(drop=True)
    # train_data = train_data.reset_index(drop=True)

    # all_data = new_df.reset_index(drop=True)

    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }

    valid_params = {'batch_size': batch_size,
                    'shuffle': False,
                    'num_workers': 0
                    }


    # print("FULL Dataset: {}".format(new_df.shape))
    # print(test_data)
    # print("TRAIN Dataset: {}".format(all_data.shape))
    # print("TEST Dataset: {}".format(test_data.shape))
    # print(test_data.iloc[:,1])

    # testing_set = SentimentData(test_data, tokenizer, MAX_LEN)

    # tp = np.asarray(all_data.iloc[:, 2], dtype=np.int64)
    # # print(tp)
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
    #
    # for kkk, (dev1, test1) in enumerate(skf.split(all_data, tp)):
    #     # 7,8
    #     # 0
    #     # 0,4,6, 8 is good
    #     # 3 is good
    #
    #     if (kkk == 1):
    #         print("group", kkk)
    # filename = f"method_name-{method_name}-dataset-{dataset}-group-{kkk}-flag-{flag}-epoches-{epoches}.npy"
    filename = f"method_name-{method_name}-dataset-{dataset}-llm-{llm}-flag-{flag}-epoches-{epoches}_v2.npy"

    logfilename = "./log2/" + filename
    # print(len(valid1))

    # dev_data = all_data.iloc[dev1, :]
    # test_data = all_data.iloc[test1, :]

    # tp_train=train_data.iloc[train1,:]
    # dev_data = dev_data.reset_index(drop=True)
    # # tp_valid = pd.DataFrame(testing_set)
    # tp_valid = test_data.reset_index(drop=True)
    # train_ds = train_ds.map(tok_fn, batched=True)
    # val_ds = val_ds.map(tok_fn, batched=True)
    # train_ds.set_format(type="torch")
    # val_ds.set_format(type="torch")

    # train_dl = torch.utils.data.DataLoader(
    #     train_ds, batch_size=args.batch_size, shuffle=True)
    # val_dl = torch.utils.data.DataLoader(
    #     val_ds, batch_size=args.batch_size)
    # Method
    if method_name.lower() == 'roberta-large'.lower():
        training_set = RobertlargeDataset(train, MAX_LEN)
        validation_set = RobertlargeDataset(val, MAX_LEN)

        dropout = 0.3
        model = RobertalargeClass(dropout, outclass)
        learning_rate = 1E-5
        # learning_rate = 1E-6
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'albert'.lower():
        training_set = AlbertClassificationDataset(dev_data, MAX_LEN)
        validation_set = AlbertClassificationDataset(tp_valid, MAX_LEN)
        dropout = 0.3
        model = AlbertCustomClassifier(dropout, outclass)
        learning_rate = 1E-5
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'roberta-base'.lower():
        training_set = RobertDataset(dev_data, MAX_LEN)
        validation_set = RobertDataset(tp_valid, MAX_LEN)
        dropout = 0.5
        model = RobertaClass(dropout, outclass)
        learning_rate = 1E-5
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'distilbert'.lower():
        training_set = distilbertDataset(dev_data, MAX_LEN)
        validation_set = distilbertDataset(tp_valid, MAX_LEN)
        dropout = 0.1
        model = distilbertClass(dropout, outclass)
        learning_rate = 1E-5
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'neobert'.lower():
        training_set = neobertDataset(dev_data, MAX_LEN)
        validation_set = neobertDataset(tp_valid, MAX_LEN)
        model = neobertClass(dropout, outclass)
        learning_rate = 1E-5
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)

    elif method_name.lower() == 'gpt2'.lower():
        training_set = gpt2Dataset(dev_data, MAX_LEN)
        validation_set = gpt2Dataset(tp_valid, MAX_LEN)
        dropout = 0.1
        model = GPT2(dropout, outclass)
        learning_rate = 1E-4
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


    elif method_name.lower() == 'Gialogpt'.lower():
        training_set = gialogptDataset(dev_data, MAX_LEN)
        validation_set = gialogptDataset(tp_valid, MAX_LEN)
        dropout = 0.3
        model = gialogpt(dropout, outclass)
        learning_rate = 1E-4
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    elif method_name.lower() == 'gpt3'.lower():
        training_set = gpt3Dataset(dev_data, MAX_LEN)
        validation_set = gpt3Dataset(tp_valid, MAX_LEN)
        dropout = 0.3
        model = gpt3(dropout, outclass)
        learning_rate = 1E-6
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)

    elif method_name.lower() == 'distilgpt2'.lower():
        training_set = GPT2distilDataset(dev_data, MAX_LEN)
        validation_set = GPT2distilDataset(tp_valid, MAX_LEN)
        model = distilgpt2(dropout, outclass)
        learning_rate = 1E-4
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 't5-small'.lower():
        training_set = T5distilDataset(dev_data, MAX_LEN)
        validation_set = T5distilDataset(tp_valid, MAX_LEN)
        model = T5Classification(dropout, outclass)
        learning_rate = 1E-3
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 't5-base'.lower():
        training_set = T5distilbaseDataset(dev_data, MAX_LEN)
        validation_set = T5distilbaseDataset(tp_valid, MAX_LEN)
        dropout = 0.3
        model = T5baseClassification(dropout, outclass)
        learning_rate = 1E-3
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'bart-base'.lower():
        training_set = bartbaseDataset(dev_data, MAX_LEN)
        validation_set = bartbaseDataset(tp_valid, MAX_LEN)
        model = bartbaseClassification(dropout, outclass)
        dropout = 0.3
        learning_rate = 1E-5
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    elif method_name.lower() == 'bart-large'.lower():
        training_set = bartlargeDataset(dev_data, MAX_LEN)
        validation_set = bartlargeDataset(tp_valid, MAX_LEN)
        model = bartlargeClassification(dropout, outclass)
        dropout = 0.5
        learning_rate = 1E-5
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    else:
        raise ValueError("method error")

        # if optim.lower()=="Adamw".lower():
        #    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
        # elif optim.lower()=="rmsprop".lower():
        #    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate)
        # elif optim.lower()=="adam".lower():
        #     optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # training_loader = DataLoader(training_set, **train_params)
    # validation_loader = DataLoader(validation_set, **valid_params)
    train_dl = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size)
    # testing_loader = DataLoader(testing_set, **test_params)
    # print(train_dl)
    # for batch in tqdm(train_dl):
    #     print(batch['ids'])
    #     break
    # # with init_empty_weights():
    model.to(device)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()

    model_to_save = copy.deepcopy(model)
    maxacc = 0.0
    maxpred1 = []
    maxtarg1 = []
    maxpred2 = []
    maxtarg2 = []

    if flag==1:
        maxacc10 = 0.0
        maxpred10_1 = []
        maxtarg10_1 = []
        maxpred10_2 = []
        maxtarg10_2 = []

        maxacc20 = 0.0
        maxpred20_1 = []
        maxtarg20_1 = []
        maxpred10_2 = []
        maxtarg10_2 = []

        maxacc30 = 0.0
        maxpred30_1 = []
        maxtarg30_1 = []
        maxpred10_2 = []
        maxtarg10_2 = []


    EPOCHS = args.epoches
    inference_time = []
    for epoch in range(EPOCHS):
        # train(model, epoch, train_dl, loss_function, optimizer, device)

        tr_loss = 0
        n_correct1 = 0
        n_correct2 = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        # for _, data in tqdm(enumerate(training_loader, 0)):
        for data in tqdm(train_dl):
            # print(data)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            # if data['token_type_ids']:
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            output1, output2 = model(ids, mask, token_type_ids)
            # else:
            #     outputs = model(ids, mask)
            targets1 = data['targets1'].to(device, dtype=torch.long)
            targets2 = data['targets2'].to(device, dtype=torch.long)
            loss = loss_function(output1, targets1) + loss_function(output2, targets2)
            # loss = loss_function(output1, targets1)

            tr_loss += loss.item()
            big_val1, big_idx1 = torch.max(output1.data, dim=1)
            n_correct1 += calcuate_accuracy(big_idx1, targets1)

            big_val2, big_idx2 = torch.max(output2.data, dim=1)
            n_correct2 += calcuate_accuracy(big_idx2, targets2)

            nb_tr_steps += 1
            nb_tr_examples += targets1.size(0)

            # if _ % 5000 == 0:
            #     loss_step = tr_loss / nb_tr_steps
            #     accu_step = (n_correct1 * 100) / nb_tr_examples
            #     accu_step = (n_correct2 * 100) / nb_tr_examples
            # print(f"Training Loss per 5000 steps: {loss_step}")
            # print(f"Training Accuracy per 5000 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # # When using GPU
            optimizer.step()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct1 * 100) / nb_tr_examples}')
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu1 = (n_correct1 * 100) / nb_tr_examples
        epoch_accu2 = (n_correct2 * 100) / nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu1}")
        print(f"Training Accuracy Epoch: {epoch_accu2}")

        # acc, pred1, targ1, pred2, targ2 = valid(model, val_dl,device = device)
        # acc = valid(model, testing_loader, 2)

        model.eval()
        n_correct1 = 0
        n_correct2 = 0
        n_wrong = 0
        total = 0
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        res_pre1, res_tar1 = [], []
        res_pre2, res_tar2 = [], []
        start = time.perf_counter()
        with torch.no_grad():
            # for _, data in tqdm(enumerate(testing_loader, 0)):
            for data in tqdm(val_dl):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)

                # if data['token_type_ids']:
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs1, outputs2 = model(ids, mask, token_type_ids)
                outputs1 = outputs1.squeeze()
                outputs2 = outputs2.squeeze()
                # else:
                # outputs = model(ids, mask).squeeze()

                # print(outputs)
                # print(targets)
                targets1 = data['targets1'].to(device, dtype=torch.long)
                targets2 = data['targets2'].to(device, dtype=torch.long)

                loss = loss_function(outputs1, targets1) + loss_function(outputs2, targets2)
                # loss = loss_function(outputs1, targets1)
                tr_loss += loss.item()
                big_val1, big_idx1 = torch.max(outputs1.data, dim=1)
                n_correct1 += calcuate_accuracy(big_idx1, targets1, flag)

                big_val2, big_idx2 = torch.max(outputs2.data, dim=1)
                n_correct2 += calcuate_accuracy(big_idx2, targets2, flag)

                for kk in range(len(targets1)):
                    res_pre1.append(big_idx1[kk].item())
                    res_tar1.append(targets1[kk].item())

                    res_pre2.append(big_idx2[kk].item())
                    res_tar2.append(targets2[kk].item())

                nb_tr_steps += 1
                nb_tr_examples += targets1.size(0)

                '''
                if _ % 5000 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples
                    print(f"Validation Loss per 100 steps: {loss_step}")
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
                '''
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu1 = (n_correct1 * 100) / nb_tr_examples
        epoch_accu2 = (n_correct2 * 100) / nb_tr_examples

        if flag == 2:
            print(f"testing Loss Epoch: {epoch_loss}")
            print(f"testing Accuracy Epoch: {epoch_accu1}")
            print(f"testing Accuracy Epoch: {epoch_accu2}")
        else:
            print(f"Validation Loss Epoch: {epoch_loss}")
            print(f"Validation Accuracy Epoch: {epoch_accu1}")
            print(f"Validation Accuracy Epoch: {epoch_accu2}")

        acc = epoch_accu1
        pred1 = res_pre1
        targ1 = res_tar1
        pred2 = res_pre2
        targ2 = res_tar2
        print(f"Accuracy on test data: {acc:.2f}")

        if acc > maxacc:
            # model_to_save = copy.deepcopy(model)
            maxacc = acc
            maxpred1 = pred1
            maxtarg1 = targ1
            maxpred2 = pred2
            maxtarg2 = targ2

        if flag==1:
            if epoch == 9:
                maxacc10 = maxacc
                maxpred10_1 = maxpred1
                maxtarg10_1 = maxtarg1
                maxpred10_2 = maxpred2
                maxtarg10_2 = maxtarg2
            elif epoch == 19:
                maxacc20 = maxacc
                maxpred20_1 = maxpred1
                maxtarg20_1 = maxtarg1
                maxpred20_2 = maxpred2
                maxtarg20_2 = maxtarg2
            elif epoch == 29:
                maxacc30 = maxacc
                maxpred30_1 = maxpred1
                maxtarg30_1 = maxtarg1
                maxpred30_2 = maxpred2
                maxtarg30_2 = maxtarg2
        end = time.perf_counter()
        inference_time.append(end - start)
          

    # acc = valid(model, testing_loader, 2)
    # acc = valid(model, testing_clips_loader, 1)
    print(f"Accuracy on test data: {maxacc:.2f}")
    # print(maxpred)
    # print(maxtarg)

    res = {
        "acc": maxacc,
        "pred1": maxpred1,
        "targ1": maxtarg1,
        "pred2": maxpred2,
        "targ2": maxtarg2,
    }


    # if flag==1:
    #     res = {
    #         "acc": maxacc,
    #         "pred1": maxpred1,
    #         "targ1": maxtarg1,
    #         "pred2": maxpred2,
    #         "targ2": maxtarg2,
    #         "acc10": maxacc10,
    #         "pred10_1": maxpred10_1,
    #         "targ10_1": maxtarg10_1,
    #         "pred10_2": maxpred10_2,
    #         "targ10_2": maxtarg10_2,
    #         "acc20": maxacc20,
    #         "pred20_1": maxpred20_1,
    #         "targ20_1": maxtarg20_1,
    #         "pred20_2": maxpred20_2,
    #         "targ20_2": maxtarg20_2,
    #         "acc30": maxacc30,
    #         "pred30_1": maxpred30_1,
    #         "targ30_1": maxtarg30_1,
    #         "pred30_2": maxpred30_2,
    #         "targ30_2": maxtarg30_2,
    #         }

    # np.save(logfilename, res)

    output_model_file = f"method_name-{method_name}-dataset-{dataset}-llm-{llm}-epoches-{epoches}_v2.bin"
    output_model = "./train2/" + output_model_file
    # output_vocab_file = './'
    print(f"Inference time: {np.mean(inference_time):.6f} seconds")
    torch.save(model_to_save.state_dict(), output_model)
    #tokenizer.save_vocabulary(output_vocab_file)



if __name__ == '__main__':

    print("Number of arguments:", len(sys.argv))
    print("Arguments are:", str(sys.argv))
    for i, arg in enumerate(sys.argv):
        print(f" {arg} ")

    parser = argparse.ArgumentParser(
        prog='Dataset',
        description='What the program does',
        epilog='Text at the bottom of help')

    parser.add_argument('--dataset_path', default='./dataset')
    # parser.add_argument('--cache_dir', default='./cache')
    parser.add_argument('--dataset', default='mamual', choices=['mamual', "pcit", "medical", "mental"])
    parser.add_argument('--llm', default='llama', choices=['llama', "mixtral", "chatgpt"])
    parser.add_argument('--method', default='bart-large')
    parser.add_argument('--lr', default=1E-5)
    parser.add_argument('--epoches', default=50, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    # parser.add_argument('--batchsize', default=5)
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--maxlen', default=512, type=int)
    parser.add_argument('--device', default='cuda:4', type=str)

    # rmsprop; adam
    parser.add_argument('--optim', default='rmsprop', type=str)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--flag', default=0, type=int)

    args = parser.parse_args()
    # "gpt2", "bart-base", "albert" ,"distilbert",
    # "roberta-large", "t5-base", "t5-small", "distilgpt2", "Gialogpt", "roberta-base",
    # for i in [ "roberta-large"]:
    # for i in [  "t5-base", "t5-small", "distilgpt2", "roberta-base", "gpt3", 'bart-large']:
    # for i in [  "albert", "distilbert", "gpt2", "Gialogpt", "bart-base"]:
    for i in [ "roberta-large"]:
        #for j in [0.0, 0.1,0.3,0.5]:
            args.method=i
            args.flag=1
            print(args)
            main(args)


    # "t5-base", "t5-small", "distilgpt2", "Gialogpt", "roberta-base", "gpt3", 't5-small', 'bart-large', "gpt2"
