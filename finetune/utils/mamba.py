import torch
import argparse
import transformers
import yaml
import os
import random

from torch.nn.utils.rnn import pad_sequence 
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer

class SFTDataset(Dataset):
   def __init__(self, data_dir, window_size=8192, stride=4096):
       super(SFTDataset, self).__init__()
       self.data = []
       for filename in os.listdir(data_dir):
           with open(os.path.join(data_dir, filename), "rb") as file:
               self.data.append(file.read())

       self.data = b''.join(self.data)
       self.window_size = window_size
       self.stride = stride

   def __len__(self):
       return (len(self.data) - self.window_size) // self.stride + 1

   def __getitem__(self, i):
       start = i * self.stride
       end = start + self.window_size
       input_ids = torch.tensor([b for b in self.data[start:end]], dtype=torch.long)
       input_ids = input_ids[:self.window_size]
       input_ids = torch.cat([input_ids, torch.zeros(self.window_size - len(input_ids), dtype=torch.long)])

          # Shift labels by one position for language model training
       labels = torch.cat([input_ids[1:], torch.tensor([-100])])
       labels = labels[:self.window_size]
       labels = torch.cat([labels, torch.zeros(self.window_size - len(labels), dtype=torch.long)])
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
    def __init__(self, data_dir: str, window_size: int = 8192, stride: int = 4096):
        self.dataset = SFTDataset(data_dir=data_dir, window_size=window_size, stride=stride)
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
