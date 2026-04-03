import os
import glob
import json
import time
import re
from datetime import datetime
import requests
from typing import Dict, List, Tuple, Any
from openai import AzureOpenAI
import importlib
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import tqdm
from llamaapi import LlamaAPI
from mistralai import Mistral
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
# oLLama API ??
OLLAMA_API_URL = ""
# OLLAMA_MODEL = "mistral"  # ???????oLLama????
# OLLAMA_MODEL = "gemma3:27b"
# OLLAMA_MODEL = "qwen2.5:14b"  # "qwen3:14b"
# OLLAMA_MODEL = "deepseek-r1:32b"
# OLLAMA_MODEL = "llama3:70b"
# OLLAMA_MODEL = "phi4:14b"

# OLLAMA_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "gpt-4o"



# ['Anxiety' 'Normal' 'Depression' 'Suicidal' 'Stress' 'Bipolar', 'Personality disorder']
class mental_dataset:
  def __init__(self):
    self.label_map = {0:'Anxiety',1: 'Normal',2: 'Depression',3: 'Suicidal',4: 'Stress',5: 'Bipolar',
                      6: 'Personality disorder'}
    self.text_map = {'Anxiety':0, 'Normal':1, 'Depression':2,'Suicidal':3,'Stress':4,'Bipolar':5,
                      'Personality disorder':6}
    self.train, self.train_label, self.test, self.test_label = self.get_dataset()

  def get_dataset(self):

    data=pd.read_csv('./dataset2/mental/Combined Data.csv')
    data = data.dropna()
    data = data.sample(n=20000 , random_state=2025).reset_index(drop=True)
    data_list=[]
    data_label=[]
    for i in range(len(data)):
      data_list.append(data.iloc[i,1])
      data_label.append(self.text_map[data.iloc[i,2]])


    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)


    for i, (train_index, test_index) in enumerate(kfold.split(data_list, data_label)):
        if i ==0:
            X_train = [data_list[i] for i in train_index]
            X_test = [data_list[i] for i in test_index]
            y_train = [data_label[i] for i in train_index]
            y_test = [data_label[i] for i in test_index]
            break

    return X_train, y_train, X_test, y_test


class medical_dataset:
  def __init__(self):
    self.label_map = self.get_label()
    self.train, self.train_label = self.get_trainset()
    self.test, self.test_label = self.get_testset()

  def get_label(self):
    df = pd.read_csv('./dataset2/medical/medical_tc_labels.csv')
    label_map={}
    for i in range(len(df)):
      label_map[df.iloc[i,0]]=df.iloc[i,1]
    return label_map

  def get_trainset(self):
    train_list=[]
    train_label=[]
    df=pd.read_csv('./dataset2/medical/medical_tc_train.csv')
    print(df.head())
    for i in range(len(df)):
      train_list.append(df.iloc[i,1])
      train_label.append(df.iloc[i,0])
    print(len(train_list))
    return train_list, train_label

  def get_testset(self):
    df=pd.read_csv('./dataset2/medical/medical_tc_test.csv')
    print(df.head())
    return df.iloc[:,1], df.iloc[:,0]


class Llama_70b_2:
    def __init__(self, args=None):
        self.API_KEY = ""
        # self.API_KEY = ""
        # self.sysmessage = "You are an advanced Dyadic Parent-Child Interaction Coding System assistant. You must need to assign a single label for the last parent speaking from the following categories: [Unlabeled Praise, Labeled Praise,Reflection, Behavior, Information Question, Descriptive Question,Indirect Command, Direct Command, Negative Talk, Neutral Talk]. Return only the selected category and nothing else."
        # self.sysmessage2 = "You are an advanced Dyadic Parent-Child Interaction Coding System assistant. You are assigning a single label for the last parent speaking. Tell me your confidence from the following level: [High, Medium, Low]. Return only the selected level and nothing else."

        self.medical_prompt = """
        You are an advanced medical-classification assistant.
        Your task is to assign one of the following condition categories for each medical abstract:
        [Neoplasms, Digestive system diseases, Nervous system diseases, Cardiovascular diseases, General pathological conditions].

        For each response, return your answer exactly in this JSON structure:
        {
          "Patient_Condition": "your_Code",
          "Confidence": your_Confidence,
          "Explanation": "your_Explanation"
        }

        - "Patient_Condition" should be the most appropriate code from the provided list.
        - "Confidence" is a float between 0.0 and 1.0, representing your confidence in the classification.
        - "Explanation" should use 1-3 sentences to justify your choice.

        Always ensure the interpretation is contextually accurate.
        """

        # self.args.llm == "llama":
        # if api:
        #     self.API_KEY = api
        # else:
        #     self.API_KEY = os.getenv('LLAMA_API_KEY')
        # self.gptmodel_name = "llama3.3-70b"
        self.gptmodel_name = "mixtral-8x22b-instruct"
        #self.gptmodel_name = "gemma3-27b"
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 5,
            "max_output_tokens": 200,
            "response_mime_type": "text/plain",
        }
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # or use another model
        self.pool = None
        self.doc_embeddings = None

    def gen_results_llama(self, prompt, max_retries=4, delay=2):
        retry_count = 0
        # prompt = None
        # # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        # if self.args.dataset=="sst2":
        #     prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        # elif self.args.dataset=="cola":
        #     prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # # print(prompt)

        api_token = self.API_KEY
        # api_token = self.args.api
        llama = LlamaAPI(api_token)

        api_request_json = {
            "model": self.gptmodel_name,
            "messages": [
                {"role": "system",
                 "content": self.medical_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.generation_config["max_output_tokens"],
            "temperature": self.generation_config["temperature"],
            "top_k": self.generation_config["top_k"],
            "top_p": self.generation_config["top_p"],
        }

        while retry_count < max_retries:
            try:
                response = llama.run(api_request_json)
                # print(output)
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                # print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1
        return


    def run(self,all_set, all_label, train_set, train_label, test_set, test_label,target_file):

        # index2label = ["UP", "LP", "RF", "BD", "IQ", "DQ", "IC", "DC", "NTA", "TA"]
        index2label = ["Neoplasms", "Digestive system diseases",
                       "Nervous system diseases", "Cardiovascular diseases",
                       "General pathological conditions"]

        mp={}
        DPICS_Code= []
        Confidence = []
        Explanation = []

        self.pool = all_set
        # print(self.pool)
        self.doc_embeddings = self.embedder.encode(self.pool, convert_to_numpy=True, show_progress_bar=True)
        # load .docx file
        # print(1)
        for i in tqdm.tqdm(range(5000, len(all_set))):
        # for i in tqdm.tqdm(range(len(test_set))):
        # for i in tqdm.tqdm(range(5000)):
            prompt=""
            prompt += "Below are a few retrieved examples:\n"

            index= self.retrieve(all_set[i],k=4)
            # print(index)
            prompt += f"1. {all_set[index[0][1]][:-2]} -> {index2label[all_label[index[0][1]]-1]}\n"
            prompt += f"2. {all_set[index[0][2]][:-2]} -> {index2label[all_label[index[0][2]]-1]}\n"
            prompt += f"3. {all_set[index[0][3]][:-2]} -> {index2label[all_label[index[0][3]]-1]}\n"
            prompt += "\n"
            prompt += "Query:\n" + all_set[i]

            respond = self.gen_results_llama(prompt)
            # print(prompt)
            # respond = respond[7:-3]
            # print(respond)

            try:
                d = json.loads(respond)
                DPICS_Code.append(d["Patient_Condition"])
                Confidence.append(d["Confidence"])
                Explanation.append(d["Explanation"])
                print(1)
            except Exception as e:
                DPICS_Code.append(None)
                Confidence.append(None)
                Explanation.append(respond)

        # mp["test_set"] = test_set
        # mp["test_label"] = test_label
        mp["MED_Code"] = DPICS_Code
        mp["Confidence"] = Confidence
        mp["Explanation"] = Explanation
        # print(DPICS_Code)
        # print(Confidence)
        # print(Explanation)

        with open(target_file, 'w') as f:
            json.dump(mp, f)
        return True

    def retrieve(self, query, k=4):


        # Compute embeddings for each document

        # print(2)
        # Dimension is determined by the embedding model, e.g., 384 for 'all-MiniLM-L6-v2'
        embedding_dim = self.doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is common for dense vectors

        # Add embeddings to the index
        index.add(self.doc_embeddings)

        # Optionally, you can map indices back to document texts

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Search the FAISS index (here we use L2 similarity)
        distances, indices = index.search(query_embedding, k)

        # Retrieve corresponding documents
        return indices
    


class mixtral:
    def __init__(self, args=None):
        self.API_KEY = ""
        # self.API_KEY = ""
        # self.sysmessage = "You are an advanced Dyadic Parent-Child Interaction Coding System assistant. You must need to assign a single label for the last parent speaking from the following categories: [Unlabeled Praise, Labeled Praise,Reflection, Behavior, Information Question, Descriptive Question,Indirect Command, Direct Command, Negative Talk, Neutral Talk]. Return only the selected category and nothing else."
        # self.sysmessage2 = "You are an advanced Dyadic Parent-Child Interaction Coding System assistant. You are assigning a single label for the last parent speaking. Tell me your confidence from the following level: [High, Medium, Low]. Return only the selected level and nothing else."
        '''
        self.medical_prompt = """
        You are an advanced medical-classification assistant.
        Your task is to assign one of the following condition categories for each medical abstract:
        [Neoplasms, Digestive system diseases, Nervous system diseases, Cardiovascular diseases, General pathological conditions].

        For each response, return your answer exactly in this JSON structure:
        {
          "Patient_Condition": "your_Code",
          "Confidence": your_Confidence,
          "Explanation": "your_Explanation"
        }

        - "Patient_Condition" should be the most appropriate code from the provided list.
        - "Confidence" is a float between 0.0 and 1.0, representing your confidence in the classification.
        - "Explanation" should use 1-3 sentences to justify your choice.

        Always ensure the interpretation is contextually accurate.
        """
        '''
        self.Mental_prompt = """
            You are an advanced Sentiment Analysis for Mental Health assistant.
            Your task is to assign one of the following condition categories for each medical statement:
            ['Anxiety' 'Normal' 'Depression' 'Suicidal' 'Stress' 'Bipolar', 'Personality disorder'].

            For each response, return your answer exactly in this JSON structure:
            {
            "Condition": "your_Code",
            "Confidence": your_Confidence,
            "Explanation": "your_Explanation"
            }

            - "Condition" should be the most appropriate code from the provided list.
            - "Confidence" is a float between 0.0 and 1.0, representing your confidence in the classification.
            - "Explanation" should use 1-3 sentences to justify your choice.

            Always ensure the interpretation is contextually accurate.
        """
        # self.args.llm == "llama":
        # if api:
        #     self.API_KEY = api
        # else:
        #     self.API_KEY = os.getenv('LLAMA_API_KEY')
        # self.gptmodel_name = "llama3.3-70b"
        self.gptmodel_name = "open-mixtral-8x22b"
        #self.gptmodel_name = "gemma3-27b"
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 5,
            "max_output_tokens": 200,
            "response_mime_type": "text/plain",
        }
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # or use another model
        self.pool = None
        self.doc_embeddings = None

    def gen_results_llama(self, prompt, max_retries=4, delay=2):
        retry_count = 0
        # prompt = None
        # # prompt = f'You are a helpful and accurate classification assistant. You will be provided with a sentence, and your task is to choose ONLY one Tag(Label_0 and Label_1) for Given Sentence from instructions. Never output others.\n\n'
        # if self.args.dataset=="sst2":
        #     prompt = self.build_prompt_sst2(group0, 0, "", df0, n) + self.build_prompt_sst2(group1, 1, text, df1, n)
        # elif self.args.dataset=="cola":
        #     prompt = self.build_prompt_cola(group0, 0, "", df0, n) + self.build_prompt_cola(group1, 1, text, df1, n)
        # # print(prompt)

        # api_token = self.API_KEY
        # api_token = self.args.api
        # llama = LlamaAPI(api_token)

        client = Mistral(api_key=self.API_KEY)

        # api_request_json = {
        #     "model": self.gptmodel_name,
        #     "messages": [
        #         {"role": "system",
        #          "content": self.medical_prompt},
        #         {"role": "user", "content": prompt},
        #     ],
        #     "max_tokens": self.generation_config["max_output_tokens"],
        #     "temperature": self.generation_config["temperature"],
        #     "top_k": self.generation_config["top_k"],
        #     "top_p": self.generation_config["top_p"],
        # }

        messages = [
            {"role": "system",
            "content": self.Mental_prompt},
            {"role": "user", "content": prompt}
        ]
        while retry_count < max_retries:
            try:
                response = client.chat.complete(
                    model=self.gptmodel_name,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.2,
                    top_p = 1.0,
                )
                response = response.choices[0].message.content
                # print(response)
                return response
            except Exception as e:
                # print(f"Error: {e}")
                time.sleep(delay)
                retry_count += 1
        return


    def run(self,all_set, all_label, train_set, train_label, test_set, test_label,target_file):

        # index2label = ["UP", "LP", "RF", "BD", "IQ", "DQ", "IC", "DC", "NTA", "TA"]
        index2label = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 
                       'Stress',  'Bipolar', 'Personality disorder']

        mp={}
        DPICS_Code= []
        Confidence = []
        Explanation = []

        self.pool = all_set
        # print(self.pool)
        self.doc_embeddings = self.embedder.encode(self.pool, convert_to_numpy=True, show_progress_bar=True)
        # load .docx file
        # print(1)
        for i in tqdm.tqdm(range(10000, len(all_set))):
        # for i in tqdm.tqdm(range(10500, len(all_set))):
        # for i in tqdm.tqdm(range(5000, 10000)):
            prompt=""
            prompt += "Below are a few retrieved examples:\n"

            index= self.retrieve(all_set[i],k=4)
            # print(index)
            prompt += f"1. {all_set[index[0][1]][:-2]} -> {index2label[all_label[index[0][1]]-1]}\n"
            prompt += f"2. {all_set[index[0][2]][:-2]} -> {index2label[all_label[index[0][2]]-1]}\n"
            prompt += f"3. {all_set[index[0][3]][:-2]} -> {index2label[all_label[index[0][3]]-1]}\n"
            prompt += "\n"
            prompt += "Query:\n" + all_set[i]

            respond = self.gen_results_llama(prompt)
            # print(prompt)
            # respond = respond[7:-3]
            # print(respond)

            try:
                d = json.loads(respond)
                DPICS_Code.append(d["Condition"])
                Confidence.append(d["Confidence"])
                Explanation.append(d["Explanation"])
                # print(1)
            except Exception as e:
                DPICS_Code.append(None)
                Confidence.append(None)
                Explanation.append(respond)

        # mp["test_set"] = test_set
        # mp["test_label"] = test_label
        mp["MED_Code"] = DPICS_Code
        mp["Confidence"] = Confidence
        mp["Explanation"] = Explanation
        # print(DPICS_Code)
        # print(Confidence)
        # print(Explanation)

        with open(target_file, 'w') as f:
            json.dump(mp, f)
        return True

    def retrieve(self, query, k=4):


        # Compute embeddings for each document

        # print(2)
        # Dimension is determined by the embedding model, e.g., 384 for 'all-MiniLM-L6-v2'
        embedding_dim = self.doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)  # L2 distance is common for dense vectors

        # Add embeddings to the index
        index.add(self.doc_embeddings)

        # Optionally, you can map indices back to document texts

        query_embedding = self.embedder.encode([query], convert_to_numpy=True)

        # Search the FAISS index (here we use L2 similarity)
        distances, indices = index.search(query_embedding, k)

        # Retrieve corresponding documents
        return indices


def main():
    # llm=Llama_70b_2()
    # medical=medical_dataset()
    llm = mixtral()
    medical = mental_dataset()
    table=medical.train
    label=medical.train_label

    y_train = medical.test
    y_test = medical.test_label
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
    # print(table[115], label[115])
    # print(table[113], label[113])
    # print(table[114], label[114])
    # print(table[600], label[600])

    # for i, (train_index, test_index) in enumerate(kfold.split(table, label)):
    #     if i ==0:
    # X_train = [table[i] for i in train_index]
    # X_test = [table[i] for i in test_index]
    # y_train = [label[i] for i in train_index]
    # y_test = [label[i] for i in test_index]

    target_file = f"mental_gpt_v4_train_end.json"

    # llm.run(table, label, X_train, y_train, X_test, y_test, target_file)
    llm.run(table, label, None, None, None, None, target_file)




if __name__ == "__main__":
    main()