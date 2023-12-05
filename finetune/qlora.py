# pip install bitsandbytes transformers peft accelerate datasets loralib sentencepiece scipy
# export MAX_JOBS=8 
# pip install flash-attn
import argparse
import yaml
import torch
import transformers
import warnings
warnings.warn = lambda *args, **kwargs: None

from datasets import load_dataset
from trl import SFTTrainer
from transformers import (TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer)
from peft import (LoraConfig, prepare_model_for_kbit_training)
from utils.tokens import add_special_tokens
from utils.prompts import chatml_prompt, vicuna_prompt
  
parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="Path to the config YAML file")
parser.add_argument("-c", "--checkpoint", type=str, default=None, required=False)
parser.add_argument("-s", "--seed", type=int, default=42, required=False)
args = parser.parse_args()

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
  
config = read_yaml_file(args.config_path)

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)
 
output_dir = "trainees/" + config["model_name"]

tokenizer = AutoTokenizer.from_pretrained(config["base_model"], padding_side = 'right', use_fast = False)
model = AutoModelForCausalLM.from_pretrained(config["base_model"], quantization_config=bnb_config, device_map={"":0}, use_flash_attention_2=True, torch_dtype=torch.bfloat16)
peft_config = LoraConfig(
    r = config["lora_rank"],
    lora_alpha = config["lora_alpha"],
    lora_dropout = config["lora_dropout"], 
    bias = config["lora_bias"],
    target_modules = config["target_modules"],
    modules_to_save = config["modules_to_save"],
    task_type="CAUSAL_LM",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

if tokenizer.pad_token == None:
    #  {'pad_token': '[PAD]', 'eos_token': '<|im_end|>', 'additional_special_tokens': ['<|im_start|>']}
    add_special_tokens({'pad_token':'[PAD]'}, tokenizer, model)
    tokenizer.save_pretrained(output_dir)

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
    peft_config = peft_config,
    args = TrainingArguments(
        per_device_train_batch_size = config["per_device_train_batch_size"],
        gradient_accumulation_steps = config["gradient_accumulation_steps"],
        warmup_steps = config["warmup_steps"],
        max_steps = config["max_steps"],
        num_train_epochs=config["num_train_epochs"],
        save_steps = config["save_steps"],
        save_total_limit = config["save_total_limit"],
        evaluation_strategy = "steps",
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        bf16_full_eval = True,
        eval_steps = config["eval_steps"],
        learning_rate = config["learning_rate"],
        bf16 = True,
        tf32 = True,
        logging_steps = config["logging_steps"],
        optim = config["optimizer"],
        weight_decay = config["weight_decay"],
        lr_scheduler_type = "linear",
        seed = args.seed,
        output_dir = output_dir,
        gradient_checkpointing=True,
        # TODO: uncomment that on the next release
        # right now it OOMs
        #gradient_checkpointing_kwargs={"use_reentrant": False},
    ),
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print_trainable_parameters(trainer.model)
trainer_stats = trainer.train(resume_from_checkpoint=args.checkpoint)
trainer.model.save_pretrained(output_dir + "/final")

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
