# https://huggingface.co/docs/transformers/perplexity
import argparse
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, type=str, help="Model dir")
parser.add_argument('-l', '--lora_dir', required=False, type=str, default='', help="Path to lora directory") 
parser.add_argument('-c', '--context', required=False, type=int, default=2048, help="Max context length")
args = parser.parse_args()
 
model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id)
 
free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}
 
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = LlamaForCausalLM.from_pretrained(model_id, device_map='auto', quantization_config=nf4_config, max_memory=max_memory)
print(model.generation_config)

if args.lora_dir != '':
    print('Loading LoRA weights')
    model = PeftModel.from_pretrained(model, args.lora_dir)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

# Define sliding window parameters
max_length = args.context
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

# Iterate over the dataset using sliding window strategy
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)
