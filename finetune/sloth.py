import argparse
import yaml
from datasets import load_dataset
from unsloth import FastLlamaModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from utils.tokens import add_special_tokens
from utils.prompts import chatml_prompt, vicuna_prompt
  
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
  
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="Path to the config YAML file")
parser.add_argument("-c", "--checkpoint", type=str, default=None, required=False)
args = parser.parse_args()
config = read_yaml_file(args.config_path)

output_dir = "trainees/" + config["model_name"]

model, tokenizer = FastLlamaModel.from_pretrained(
    model_name = config["base_model"],
    max_seq_length = config["max_len"],
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

if tokenizer.pad_token == None:
    add_special_tokens({'pad_token':'[PAD]'}, tokenizer, model)
    tokenizer.save_pretrained(output_dir)

model = FastLlamaModel.get_peft_model(
    model,
    r = config["lora_rank"],
    target_modules = config["target_modules"],
    lora_alpha = config["lora_alpha"],
    lora_dropout = 0, # Currently only supports dropout = 0
    bias = "none",    # Currently only supports bias = "none"
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = config["max_len"],
)

dataset = load_dataset(config["dataset"], split = "train")
dataset = dataset.train_test_split(test_size=config["dataset_test_split"])
train_dataset = dataset['train'].shuffle()
eval_dataset = dataset['test']

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    max_seq_length = config["max_len"],
    packing = True,
    formatting_func = vicuna_prompt,
    args = TrainingArguments(
        per_device_train_batch_size = config["per_device_train_batch_size"],
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        warmup_steps = config["warmup_steps"],
        max_steps = config["max_steps"],
        num_train_epochs=config["num_train_epochs"],
        save_steps = config["save_steps"],
        save_total_limit = config["save_total_limit"],
        learning_rate = config["learning_rate"],
        bf16 = True,
        tf32 = True,
        evaluation_strategy = "steps",
        per_device_eval_batch_size=1,
        bf16_full_eval = True,
        eval_steps = config["eval_steps"],
        logging_steps = config["logging_steps"],
        optim = config["optimizer"],
        weight_decay = config["weight_decay"],
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train(resume_from_checkpoint=args.checkpoint)
trainer.save_model(output_dir + "/final")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
