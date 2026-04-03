import tqdm
import pandas as pd
import numpy as np
import os, json
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold



class mamual:
    def __init__(self):
        # self.train, self.train_label = self.get_trainset()
        # self.test, self.test_label = self.get_testset()
        self.definition = [
            "An Unlabeled Praise provides a positive evaluation of the child, an attribute of the child, or a nonspecific activity, behavior, or product of the child.",
            "Labeled Praise provides a positive evaluation of a specific attribute, product, or behavior of the child.",
            "A Reflection is a declarative phrase or statement that has the same meaning as the child's verbalization. The reflection may repeat, paraphrase, or elaborate upon the child's verbalization but may not change the meaning of the child's statement or interpret unstated ideas.",
            "Behavior Descriptions are non-evaluative, declarative sentences or phrases in which the subject is the other person, and the verb describes that person's ongoing or immediately completed (< 5 sec.) observable verbal or nonverbal behavior.",
            "Questions that request specific information from the child other than a brief response (e.g., yes, no, maybe) are Information Questions, even if the child gives a brief response, such as 'I don't know,' or no response at all.",
            "A Descriptive Question is a descriptive or reflective comment or statement expressed in question form that requires no more than a brief affirmative or negative response (e.g., 'yes' or 'no'), even if the child gives additional information in response or does not respond.",
            "An Indirect Command is a suggestion for a vocal or motor behavior or a mental or internal, unobservable action to be performed that is stated in question form or such that it is unclear if the child must complete the request.",
            "Direct commands are declarative statements that contain an order or direction for a vocal or motor behavior, or a mental or internal, unobservable action to be performed and indicate that the child is to perform this behavior.",
            "Negative Talk is a verbal expression of disapproval of the child or the child's attributes, activities, products, or choices. Negative Talk also includes sassy, sarcastic, rude, or impudent speech.",
            "Neutral talk statements introduce information about other people, objects, events, or activities, or simply acknowledge current activity, but do not direct, describe or evaluate the child's current or immediately completed behavior."
        ]
        self.index2label = ["Unlabeled Praise", "Labeled Praise",
                              "Reflection", "Behavior", "Information Question", "Descriptive Question",
                              "Indirect Command", "Direct Command",
                              "Negative Talk", "Neutral Talk"]
        # self.label2index = {"Neoplasms": 0, "Digestive system diseases": 1,
        #                     "Nervous system diseases": 2, "Cardiovascular diseases": 3,
        #                     "General pathological conditions": 4}

    def get_llama_results(self):

        data, label = self.load_jsonl('./clean/mamual_gpt.json', flag="train", start=0)
        print(data[0])
        print(label[0])
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 7:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_mixtral_results(self):
        data, label = self.load_jsonl('./clean/mamual_gpt2.json', flag="train", start=0)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 0:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)
        # return train_ds, val_ds

    def get_chatgpt4o_results(self):
        data, label = self.load_jsonl('./clean/mamual_gpt3.json', flag="train", start=0)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 4:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break
        # print(train_ds)
        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

            # print(tp_prompt)

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def load_jsonl(self, path, flag=None, start=0):

        with open(path, 'r') as file:
            dialogue = json.load(file)

        rows = []
        label = []
        for i in range(len(dialogue['DPICS_Code'])):
            if dialogue["DPICS_Code"][i] is not None:
                tp_prompt = ""
                tp_prompt += f"Dialogue: {dialogue['test_set'][i]}\n"
                tp_prompt += f"AI-Code: {self.index2label[dialogue['DPICS_Code'][i]]}\n"
                tp_prompt += f"Explanation: {dialogue['Explanation'][i]}\n"
                # print(dialogue['DPICS_Code'][i])
                tp_prompt += f"Definition: {self.definition[dialogue['DPICS_Code'][i]]}"

                if dialogue['DPICS_Code'][i] == dialogue['test_label'][i]:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Correct",
                                 "labels2": self.index2label[int(dialogue['test_label'][i])]
                                 })

                else:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Error",
                                 "labels2": self.index2label[int(dialogue['test_label'][i])]
                                 })
                label.append(int(dialogue['test_label'][i]))
        print(len(rows), len(label))
        return rows, label


class pcit:
    def __init__(self):
        # self.train, self.train_label = self.get_trainset()
        # self.test, self.test_label = self.get_testset()
        self.definition = [
            "An Unlabeled Praise provides a positive evaluation of the child, an attribute of the child, or a nonspecific activity, behavior, or product of the child.",
            "Labeled Praise provides a positive evaluation of a specific attribute, product, or behavior of the child.",
            "A Reflection is a declarative phrase or statement that has the same meaning as the child's verbalization. The reflection may repeat, paraphrase, or elaborate upon the child's verbalization but may not change the meaning of the child's statement or interpret unstated ideas.",
            "Behavior Descriptions are non-evaluative, declarative sentences or phrases in which the subject is the other person, and the verb describes that person's ongoing or immediately completed (< 5 sec.) observable verbal or nonverbal behavior.",
            "Questions that request specific information from the child other than a brief response (e.g., yes, no, maybe) are Information Questions, even if the child gives a brief response, such as 'I don't know,' or no response at all.",
            "A Descriptive Question is a descriptive or reflective comment or statement expressed in question form that requires no more than a brief affirmative or negative response (e.g., 'yes' or 'no'), even if the child gives additional information in response or does not respond.",
            "An Indirect Command is a suggestion for a vocal or motor behavior or a mental or internal, unobservable action to be performed that is stated in question form or such that it is unclear if the child must complete the request.",
            "Direct commands are declarative statements that contain an order or direction for a vocal or motor behavior, or a mental or internal, unobservable action to be performed and indicate that the child is to perform this behavior.",
            "Negative Talk is a verbal expression of disapproval of the child or the child's attributes, activities, products, or choices. Negative Talk also includes sassy, sarcastic, rude, or impudent speech.",
            "Neutral talk statements introduce information about other people, objects, events, or activities, or simply acknowledge current activity, but do not direct, describe or evaluate the child's current or immediately completed behavior."
        ]
        self.index2label = ["Unlabeled Praise", "Labeled Praise",
                  "Reflection", "Behavior", "Question",
                  "Command", "Negative Talk", "Neutral Talk"]
        # self.label2index = {"Neoplasms": 0, "Digestive system diseases": 1,
        #                     "Nervous system diseases": 2, "Cardiovascular diseases": 3,
        #                     "General pathological conditions": 4}

    def get_llama_results(self):

        data, label = self.load_jsonl('./clean/pcit_gpt_v2.json', flag="train", start=0)
        print(data[0])
        print(label[0])
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 3:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_mixtral_results(self):
        data, label = self.load_jsonl('./clean/pcit_gpt_v4.json', flag="train", start=0)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 7:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_chatgpt4o_results(self):
        data, label = self.load_jsonl('./clean/pcit_gpt_v6.json', flag="train", start=0)

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)

        for i, (train_index, test_index) in enumerate(kfold.split(data, label)):
            if i == 1:
                train_ds = [data[i] for i in train_index]
                val_ds = [data[i] for i in test_index]
                break

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def load_jsonl(self, path, flag=None, start=0):

        with open(path, 'r') as file:
            dialogue = json.load(file)

        rows = []
        label = []
        for i in range(len(dialogue['DPICS_Code'])):
            if dialogue["DPICS_Code"][i] is not None:
                tp_prompt = ""
                tp_prompt += f"Dialogue: {dialogue['test_set'][i]}\n"
                tp_prompt += f"AI-Code: {self.index2label[dialogue['DPICS_Code'][i]]}\n"
                tp_prompt += f"Explanation: {dialogue['Explanation'][i]}\n"
                # print(dialogue['DPICS_Code'][i])
                tp_prompt += f"Definition: {self.definition[dialogue['DPICS_Code'][i]]}"

                if dialogue['DPICS_Code'][i] == dialogue['test_label'][i]:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Correct",
                                 "labels2": self.index2label[int(dialogue['test_label'][i])]
                                 })
                else:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Error",
                                 "labels2": self.index2label[int(dialogue['test_label'][i])]
                                 })
                label.append(int(dialogue['test_label'][i]))
        print(len(rows), len(label))
        return rows, label


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
        train1_ds = self.load_jsonl('./clean/med_gpt_v2_train5000_clean.json', flag="train", start = 0)
        train2_ds = self.load_jsonl('./clean/med_gpt_v2_trainend_clean.json', flag="train", start=5000)
        val_ds = self.load_jsonl('./clean/med_gpt_v2_test_clean.json', flag='test')
        train_ds = train1_ds + train2_ds

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_mixtral_results(self):
        train1_ds = self.load_jsonl('./clean/med_gpt_v4_train5000_clean.json', flag="train", start = 0)
        train2_ds = self.load_jsonl('./clean/med_gpt_v4_train10500_clean.json', flag="train", start=5000)
        train3_ds = self.load_jsonl('./clean/med_gpt_v4_trainend_clean.json', flag="train", start=10500)
        val_ds = self.load_jsonl('./clean/med_gpt_v4_test_clean.json', flag='test')
        train_ds = train1_ds + train2_ds + train3_ds

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_chatgpt4o_results(self):
        train_ds = self.load_jsonl('./clean/med_gpt_v6_train_clean.json', flag="train", start = 0)
        val_ds = self.load_jsonl('./clean/med_gpt_v6_test_clean.json', flag='test', start = 0)

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def load_jsonl(self, path, flag=None, start = 0):
        if flag == "train":
            data = self.train[start:]
            data_label = self.train_label[start:]
            with open(path, 'r') as file:
                dialogue = json.load(file)
        elif flag == "test":
            data = self.test[start:]
            data_label = self.test_label[start:]
            with open(path, 'r') as file:
                dialogue = json.load(file)

        rows = []
        for i in range(len(dialogue['MED_Code'])):
            if dialogue["MED_Code"][i] is not None and dialogue['MED_Code'][i] in self.label2index.keys():
              
                tp_prompt = ""
                tp_prompt += f"Dialogue: {data[i]}\n"
                tp_prompt += f"AI-Code: {dialogue['MED_Code'][i]}\n"
                tp_prompt += f"Explanation: {dialogue['Explanation'][i]}\n"
                # tp_prompt += f"Definition: {self.definition[self.label2index[dialogue['MED_Code'][i]]]}"

                if self.label2index[dialogue['MED_Code'][i]] == data_label[i] - 1:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Correct",
                                 "labels2": self.index2label[int(data_label[i] - 1)]
                                 })
                else:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Error",
                                 "labels2": self.index2label[int(data_label[i] - 1)]
                                 })
        print(len(rows))
        # print(rows[0])
        return rows

class mental_dataset:
    def __init__(self):
        self.label_map = {0:'Anxiety',1: 'Normal',2: 'Depression',3: 'Suicidal',4: 'Stress',5: 'Bipolar',
                          6: 'Personality disorder'}
        self.text_map = {'Anxiety':0, 'Normal':1, 'Depression':2,'Suicidal':3,'Stress':4,'Bipolar':5,
                          'Personality disorder':6}
        self.definition = [
            "Anxiety in mental health is a persistent feeling of worry, fear, or unease that can interfere with daily functioning.",
            "Normal refers to thoughts, emotions, and behaviors that allow a person to function effectively in daily life and adapt to social and cultural expectations without significant distress.",
            "Depression is a common mental health disorder characterized by persistent feelings of sadness, hopelessness, and a loss of interest or pleasure in daily activities.",
            "Suicidal refers to having thoughts, plans, or intentions of ending one's own life.",
            "Stress is the body's and mind's response to perceived challenges or threats, causing emotional and physical tension.",
            "Bipolar disorder is a mental health condition characterized by extreme mood swings that include emotional highs (mania or hypomania) and lows (depression).",
            "A personality disorder is a type of mental health condition characterized by long-term patterns of thinking, feeling, and behaving that are inflexible and deviate significantly from cultural expectations, causing distress or impaired functioning."
        ]
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


    def get_llama_results(self):
        train1_ds = self.load_jsonl('./clean/mental_gpt_v2_train_8000_clean.json', flag="train", start = 0)
        train2_ds = self.load_jsonl('./clean/mental_gpt_v2_train_end_clean.json', flag="train", start=8000)
        val_ds = self.load_jsonl('./clean/mental_gpt_v2_test_clean.json', flag='test')
        train_ds = train1_ds + train2_ds

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_mixtral_results(self):
        pass
        train1_ds = self.load_jsonl('./clean/mental_gpt_v4_train_5000_clean.json', flag="train", start = 0)
        train2_ds = self.load_jsonl('./clean/mental_gpt_v4_train_10000_clean.json', flag="train", start=5000)
        train3_ds = self.load_jsonl('./clean/mental_gpt_v4_train_end_clean.json', flag="train", start=10000)
        val_ds = self.load_jsonl('./clean/mental_gpt_v4_test_clean.json', flag='test')
        train_ds = train1_ds + train2_ds + train3_ds

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def get_chatgpt4o_results(self):
        train1_ds = self.load_jsonl('./clean/mental_gpt_v6_train_10000_clean.json', flag="train", start = 0)
        train2_ds = self.load_jsonl('./clean/mental_gpt_v6_train_end_clean.json', flag="train", start=10000)
        val_ds = self.load_jsonl('./clean/mental_gpt_v6_test_clean.json', flag='test', start = 0)
        train_ds = train1_ds + train2_ds

        for i in range(len(train_ds)):
            tp_prompt = train_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{train_ds[i]['labels1']}", "Correct_Condition": "{train_ds[i]['labels2']}"}}"""
            train_ds[i]['gt'] = tp_prompt

        for i in range(len(val_ds)):
            tp_prompt = val_ds[i]['text']
            tp_prompt += f"\n\n###"
            tp_prompt += f"""{{"Judge": "{val_ds[i]['labels1']}", "Correct_Condition": "{val_ds[i]['labels2']}"}}"""
            val_ds[i]['gt'] = tp_prompt

        return Dataset.from_list(train_ds), Dataset.from_list(val_ds)

    def load_jsonl(self, path, flag=None, start = 0):
        if flag == "train":
            data = self.train[start:]
            data_label = self.train_label[start:]
            with open(path, 'r') as file:
                dialogue = json.load(file)
        elif flag == "test":
            data = self.test[start:]
            data_label = self.test_label[start:]
            with open(path, 'r') as file:
                dialogue = json.load(file)

        rows = []
        for i in range(len(dialogue['MED_Code'])):
            if dialogue["MED_Code"][i] is not None and dialogue['MED_Code'][i] in self.text_map.keys():
                tp_prompt = ""
                tp_prompt += f"Dialogue: {data[i]}\n"
                tp_prompt += f"AI-Code: {dialogue['MED_Code'][i]}\n"
                tp_prompt += f"Explanation: {dialogue['Explanation'][i]}\n"
                # print(dialogue['MED_Code'][i])
                # tp_prompt += f"Definition: {self.definition[self.text_map[dialogue['MED_Code'][i]]]}"

                if self.text_map[dialogue['MED_Code'][i]] == data_label[i]:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Correct",
                                 "labels2": self.label_map[int(data_label[i])]
                                 })
                else:
                    rows.append({"text": tp_prompt,
                                 "labels1": "Error",
                                 "labels2": self.label_map[int(data_label[i])]
                                 })
        print(len(rows))
        print(rows[0])
        return rows



if __name__ == "__main__":
    # med = mental_dataset()
    med = medical_dataset()
    # med = mamual()
    # med = pcit()
    # x = med.get_chatgpt4o_results()
    x = med.get_llama_results()
    print(x)