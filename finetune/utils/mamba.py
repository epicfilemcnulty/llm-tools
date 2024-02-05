import torch
import argparse
import transformers
import yaml
import os
import random

from torch.nn.utils.rnn import pad_sequence 
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer

@dataclass
class DatasetConfig:
    data_dir: str
    window_size: int = 8192
    stride: int = 4096
    sod_token: bytes = b'<sod>'
    eod_token: bytes = b'<eod>'

class SFTDataset(Dataset):
   def __init__(self, config):
       super(SFTDataset, self).__init__()
       self.config = config
       self.data = []

       file_names_parquet=[]
       file_names_txt=[]

       for filename in os.listdir(config.data_dir):
           if filename.endswith('.parquet'):
               file_names_parquet.append(filename)
           elif filename.endswith('.txt'):
               file_names_txt.append(filename)
       dfs=[]
       for filename in file_names_parquet:
           df=pd.read_parquet(os.path.join(config.data_dir,filename))
           dfs.append(df)
       # Combine all DataFrames into a single DataFrame and shuffle it.
       df_combined=pd.concat(dfs,axis=0).reset_index(drop=True)
       df_combined=df_combined.sample(frac=1).reset_index(drop=True)

       for index,row in df_combined.iterrows():
           text=row['text'].encode('utf-8')
           text=self.config.sod_token + text + self.config.eod_token
           self.data.append(text)
       # Process the rest of the txt files.
       random.shuffle(file_names_txt)

       for filename in file_names_txt:
           with open(os.path.join(config.data_dir,filename),"rb") as file:
               text = file.read()
               self.data.append(self.config.sod_token + text + self.config.eod_token)

       self.data = b''.join(self.data)

   def __len__(self):
       return (len(self.data) - self.config.window_size) // self.config.stride + 1

   def __getitem__(self, i):
       start = i * self.config.stride
       end = start + self.config.window_size
       input_ids = torch.tensor([b for b in self.data[start:end]], dtype=torch.long)
       input_ids = input_ids[:self.config.window_size]
       input_ids = torch.cat([input_ids, torch.zeros(self.config.window_size - len(input_ids), dtype=torch.long)])

          # Shift labels by one position for language model training
       labels = torch.cat([input_ids[1:], torch.tensor([-100])])
       labels = labels[:self.config.window_size]
       labels = torch.cat([labels, torch.zeros(self.config.window_size - len(labels), dtype=torch.long)])
       return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSFTDataset(object):

    def __call__(self, instances):

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  
        return {                                                                                   
            'input_ids': input_ids,                                                        
            'attention_mask': (input_ids != 0),                                            
            'labels': labels,                                                              
        }  
    
class ByteDataModule():
    def __init__(self, config: DatasetConfig):
        self.dataset = SFTDataset(config)
        self.data_collator = DataCollatorForSFTDataset()

class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    def save_model(self, output_dir, _internal_call=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
