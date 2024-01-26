import torch
import argparse
from transformers import AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sys
import json
import time
from tqdm import tqdm

def tokenize(prompt):
    return [b for b in prompt.encode()]

def detokenize(output_ids):                                                               
    return bytes(output_ids[0]).decode(errors='ignore')

def run_mamba(model, prompt, max_new_tokens):

    input_ids = torch.LongTensor(tokenize(prompt)).unsqueeze(0).cuda()
    
    out = model.generate(
        input_ids=input_ids,
        max_length=len(input_ids)+max_new_tokens,
        eos_token_id=0
    )
    return detokenize(out)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_dir", help="Path to the model dir")
parser.add_argument("-n", "--new_tokens", type=int, default=256, required=False)
args = parser.parse_args()
model = MambaLMHeadModel.from_pretrained(args.model_dir, device="cuda", dtype=torch.bfloat16)

print("="*80)
while True:

    prompt = input("Prompt > ")
    
    answer = run_mamba(model, prompt, args.new_tokens)
    print(answer)
    print("="*80)
    print("")


