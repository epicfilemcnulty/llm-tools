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
    def __init__(self, data_dir, chunk_size=8192, shuffle=True):
        super(SFTDataset, self).__init__()
        self.data = []
        for filename in os.listdir(data_dir):
            with open(os.path.join(data_dir, filename), "rb") as file:
                self.data.append(file.read())

        if shuffle:
            # Splitting data into chunks of size chunk_size
            chunks = [self.data[i:i + chunk_size] for i in range(0, len(self.data), chunk_size)]
            random.shuffle(chunks)
            self.data = b''.join([b''.join(chunk) for chunk in chunks])
        else:
            self.data = b''.join(self.data)

        self.chunk_size = chunk_size 
                                                                                                    
    def __len__(self):                                                                              
        return len(self.data) // self.chunk_size                                                    
                                                                                                    
    def __getitem__(self, i):                                                                       
        start = i * (self.chunk_size)                                                           
        end = (i + 1) * (self.chunk_size)                                                       
        input_ids = torch.tensor([b for b in self.data[start:end]], dtype=torch.long)
        # Shift labels by one position for language model training                                  
        labels = torch.cat([input_ids[1:], torch.tensor([-100])])
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
    def __init__(self, data_dir: str, chunk_size: int, shuffle: bool = True):

        self.dataset = SFTDataset(data_dir=data_dir, chunk_size=chunk_size, shuffle=shuffle)
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
