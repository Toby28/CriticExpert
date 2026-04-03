from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, DistilBertTokenizerFast, DistilBertModel
import torch
import torch.nn as nn

'''
class RobertlargeDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
        self.data = dataframe
        self.text = dataframe.Phrase
        self.targets1 = self.data.Sentiment1
        self.targets2 = self.data.Sentiment2
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        # encoded = self.tokenizer.encode(text, truncation=True)
        # print(text)
        # print(len(encoded))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }
'''

class RobertlargeDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
        self.data = dataframe
        self.text = dataframe["text"]
        self.targets1 = self.data["labels1"]
        self.targets2 = self.data["labels2"]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        '''
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True, 
            return_token_type_ids=True
        )
        '''
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',           # ? Modern version
            truncation=True,                # ? Ensure truncation if text > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'             # ? Return as PyTorch tensors
        )
                
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        # encoded = self.tokenizer.encode(text, truncation=True)
        # print(text)
        # print(len(encoded))
        '''
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }
        '''
        return {
            'ids': torch.tensor(ids.squeeze(0), dtype=torch.long),
            'mask': torch.tensor(mask.squeeze(0), dtype=torch.long),
            'token_type_ids':  torch.tensor(token_type_ids.squeeze(0), dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }
        
class RobertalargeClass(torch.nn.Module):
    def __init__(self, dropout,outclass):
        super(RobertalargeClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-large")

        # Resize pos embeddings manually
        '''
        old_embeddings = self.l1.embeddings.position_embeddings
        old_max_pos, dim = old_embeddings.weight.shape
        new_pos_embed = nn.Embedding(1024, dim)
        new_pos_embed.weight.data[:old_max_pos] = old_embeddings.weight.data
        self.l1.embeddings.position_embeddings = new_pos_embed
        self.l1.config.max_position_embeddings = 1024
        '''
        self.pre_classifier1 = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier1 = torch.nn.Linear(1024, 2)
        
        self.pre_classifier2 = torch.nn.Linear(1024, 1024)
        self.classifier2 = torch.nn.Linear(1024, outclass)

    def forward(self, input_ids, attention_mask, token_type_ids):
        #print(input_ids.size())
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #print(output_1)
        hidden_state = output_1[0]
        #print(hidden_state.size())
        pooler = hidden_state[:, 0]
        #print(pooler.size())
        
        pooler2 = self.pre_classifier2(pooler)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)
        output2 = self.classifier2(pooler2)
        
        pooler1 = self.pre_classifier1(pooler2)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)
        output1 = self.classifier1(pooler1)
        

        
        
        return output1, output2


class RobertDataset(Dataset):
    def __init__(self, dataframe, max_len):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
        self.data = dataframe
        self.text = dataframe.Phrase
        self.targets1 = self.data.Sentiment1
        self.targets2 = self.data.Sentiment2
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        # encoded = self.tokenizer.encode(text, truncation=True)
        # print(text)
        # print(len(encoded))

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }


class RobertaClass(torch.nn.Module):
    def __init__(self, dropout,outclass):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier1 = torch.nn.Linear(384, 384)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier1 = torch.nn.Linear(384, 2)
        
        self.pre_classifier2 = torch.nn.Linear(768, 384)
        self.classifier2 = torch.nn.Linear(384, outclass)

    def forward(self, input_ids, attention_mask, token_type_ids):
        #print(input_ids.size())
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        #print(output_1)
        hidden_state = output_1[0]
        #print(hidden_state.size())
        pooler = hidden_state[:, 0]
        #print(pooler.size())
        pooler2 = self.pre_classifier2(pooler)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)
        output2 = self.classifier2(pooler2)
        
        pooler1 = self.pre_classifier1(pooler2)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)
        output1 = self.classifier1(pooler1)
        
        return output1, output2



class distilbertDataset(Dataset):
    def __init__(self, dataframe, max_len):
        # self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
        self.tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', truncation=True, do_lower_case=True)
        self.data = dataframe
        self.text = dataframe["text"]
        self.targets1 = self.data["labels1"]
        self.targets2 = self.data["labels2"]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        '''
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True, 
            return_token_type_ids=True
        )
        '''
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_len,
            return_token_type_ids=True
        )
                
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        # print(ids.shape, mask.shape)
        # encoded = self.tokenizer.encode(text, truncation=True)
        # print(text)
        # print(len(encoded))
        '''
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }
        '''
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids':  torch.tensor(token_type_ids, dtype=torch.long),
            'targets1': torch.tensor(self.targets1[index], dtype=torch.float),
            'targets2': torch.tensor(self.targets2[index], dtype=torch.float)
        }
        
class distilbertClass(torch.nn.Module):
    def __init__(self, dropout,outclass):
        super(distilbertClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("distilroberta-base")

        # Resize pos embeddings manually
        '''
        old_embeddings = self.l1.embeddings.position_embeddings
        old_max_pos, dim = old_embeddings.weight.shape
        new_pos_embed = nn.Embedding(1024, dim)
        new_pos_embed.weight.data[:old_max_pos] = old_embeddings.weight.data
        self.l1.embeddings.position_embeddings = new_pos_embed
        self.l1.config.max_position_embeddings = 1024
        '''
        self.pre_classifier1 = torch.nn.Linear(384, 384)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier1 = torch.nn.Linear(384, 2)
        
        self.pre_classifier2 = torch.nn.Linear(768, 384)
        self.classifier2 = torch.nn.Linear(384, outclass)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        #print(input_ids.size())
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        #print(output_1)
        hidden_state = output_1[0]
        #print(hidden_state.size())
        pooler = hidden_state[:, 0]
        #print(pooler.size())
        
        pooler2 = self.pre_classifier2(pooler)
        pooler2 = torch.nn.ReLU()(pooler2)
        pooler2 = self.dropout(pooler2)
        output2 = self.classifier2(pooler2)
        
        pooler1 = self.pre_classifier1(pooler2)
        pooler1 = torch.nn.ReLU()(pooler1)
        pooler1 = self.dropout(pooler1)
        output1 = self.classifier1(pooler1)
        
        return output1, output2
