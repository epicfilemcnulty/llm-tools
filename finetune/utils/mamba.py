import torch
import argparse
import transformers
import yaml
import os
import random

from torch.nn.utils.rnn import pad_sequence 
from tqdm.auto import tqdm
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer

tqdm.pandas()

@dataclass
class ByteDatasetConfig:
    data_dir: str
    window_size: int = 8192
    stx_token: bytes = b'\x02' # STX ASCII control character (Start of TeXt)
    etx_token: bytes = b'\x03' # ETX ASCII control character (End of TeXt)


class ByteDataset(Dataset):

   def calc_chunks(self, text):
       if len(text) == 0:
           return -1
       delimiters = len(self.config.stx_token) + len(self.config.etx_token)
       length = len(text) + delimiters
       chunks = length // self.config.window_size
       rest_bytes = length - chunks * self.config.window_size
       if rest_bytes > 0:
           chunks += 1
       return chunks

   def __init__(self, config):
       super(ByteDataset, self).__init__()
       self.config = config
       self.data = []
       self.chunks = []

       file_names_parquet=[]
       file_names_txt=[]

       for filename in os.listdir(config.data_dir):
           if filename.endswith('.parquet'):
               file_names_parquet.append(filename)
           elif filename.endswith('.txt'):
               file_names_txt.append(filename)

       print("Reading and combining parquet files")
       dfs=[]
       for filename in tqdm(file_names_parquet):
           df=pd.read_parquet(os.path.join(config.data_dir,filename))
           dfs.append(df)
       # Combine all DataFrames into a single DataFrame and shuffle it.
       if len(dfs) > 0:
           df_combined=pd.concat(dfs,axis=0).reset_index(drop=True)
           df_combined=df_combined.sample(frac=1).reset_index(drop=True)

           print("Processing parquet dataframes")
           for index,row in tqdm(df_combined.iterrows(), total=df_combined.shape[0]):
               text=row['text'].encode('utf-8')
               chunks = self.calc_chunks(text)
               if chunks > 0:
                   self.data.append(text)
                   self.chunks.append(chunks)

       # Process the rest of the txt files.
       random.shuffle(file_names_txt)

       for filename in file_names_txt:
           with open(os.path.join(config.data_dir,filename),"rb") as file:
               text = file.read()
               chunks = self.calc_chunks(text)
               if chunks > 0:
                   self.data.append(text)
                   self.chunks.append(chunks)

   def __len__(self):
       return sum(self.chunks)

   def map_idx(self, i):
      accum_chunks = 0
      for idx, num_chunks in enumerate(self.chunks):
          if accum_chunks + num_chunks > i:
              return idx, i - accum_chunks
          accum_chunks += num_chunks
      raise ValueError(f"Index {i} out of range")

   def __getitem__(self, i):
       idx, offset = self.map_idx(i)
       start = offset * self.config.window_size
       end = start + self.config.window_size

       if offset == 0:
           input_ids = [b for b in self.config.stx_token]
           input_ids += [b for b in self.data[idx][start:end-len(self.config.stx_token)]]
       else:
           input_ids = [b for b in self.data[idx][start-len(self.config.stx_token):end]]

       # Check if it's the last chunk of a document.
       if (offset + 1) * self.config.window_size >= len(self.data[idx]):
           input_ids += [b for b in self.config.etx_token]

       input_ids_tensor = torch.tensor(input_ids[:self.config.window_size], dtype=torch.long)
       # Pad sequence to desired length.
       padded_input_ids = torch.cat([input_ids_tensor, torch.zeros(self.config.window_size - len(input_ids_tensor), dtype=torch.long)])

       # Shift labels by one position for language model training
       labels_tensor = torch.cat([padded_input_ids[1:],torch.tensor([-100])])
       padded_labels = torch.cat([labels_tensor, torch.zeros(self.config.window_size - len(labels_tensor), dtype=torch.long)])
       return dict(input_ids=padded_input_ids, labels=padded_labels)

@dataclass
class DataCollatorForByteDataset(object):

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
    def __init__(self, config: ByteDatasetConfig):
        self.dataset = ByteDataset(config)
        self.data_collator = DataCollatorForByteDataset()

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
