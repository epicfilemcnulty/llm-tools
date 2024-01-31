import torch
import argparse
import transformers
import os
import random

import bitsandbytes as bnb

from tqdm import tqdm
from transformers import TrainingArguments
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MambaConfig
from utils.mamba import MambaTrainer, ByteDataModule
from utils.misc import read_yaml_file

def run(args):
    
    config = read_yaml_file(args.config_path)

    if "base_model" in config:
        model_dir = config["base_model"]
        model = MambaLMHeadModel.from_pretrained(model_dir, device="cuda", dtype=torch.bfloat16)
    else:
        model_config = MambaConfig(
          d_model=config["d_model"],
          n_layer=config["n_layer"],
          vocab_size=256,  # Set vocab size to 256 for byte-level inputs.
        )
        model = MambaLMHeadModel(config=model_config, device="cuda", dtype=torch.bfloat16)

    train_data = ByteDataModule(
        data_dir=config["dataset"],
        window_size=config["chunk_size"],
        stride=config["stride"],
    )
    adam = bnb.optim.Adam8bit(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95))
    train_dataset = train_data.dataset
    output_dir = "trainees/" + config["model_name"]

    trainer = MambaTrainer(
        model=model,
        train_dataset=train_dataset,
        args=TrainingArguments(
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_grad_norm = config["max_grad_norm"],
            warmup_steps = config["warmup_steps"],
            max_steps = config["max_steps"],
            num_train_epochs=config["num_train_epochs"],
            #evaluation_strategy = "steps",
            #eval_steps = config["eval_steps"],
            save_steps = config["save_steps"],
            save_total_limit = config["save_total_limit"],
            logging_steps=config["logging_steps"],
            lr_scheduler_type = config["scheduler"],
            bf16 = True,
            tf32 = True,
            seed=args.seed,
            output_dir=output_dir,
        ),
        optimizers=(adam, None),
        data_collator=train_data.data_collator,
    )
    trainer.train(resume_from_checkpoint=args.checkpoint)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, required=False)
    parser.add_argument("-s", "--seed", type=int, default=42, required=False)
    args = parser.parse_args()
    run(args)
