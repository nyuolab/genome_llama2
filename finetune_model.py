import torch
import os
import random
import numpy as np
from model import GenomeLlama2ForClassification
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from util import ConfigDict
import logging
import time
import argparse
import yaml

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_finetune(finetune_config):
    torch.set_float32_matmul_precision('medium')

    print(f"Finetune Config: {finetune_config}")

    tokenizer = AutoTokenizer.from_pretrained(finetune_config.TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = GenomeLlama2ForClassification(tokenizer, finetune_config)

    # Initialize with a seed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(finetune_config.SEED)
    np.random.seed(finetune_config.SEED)
    torch.manual_seed(finetune_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(finetune_config.SEED)

    # Define lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=finetune_config.GPUS,
        max_steps=finetune_config.STEPS,
        precision=finetune_config.PRECISION,
        accumulate_grad_batches=finetune_config.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=finetune_config.GRADIENT_CLIP_VAL,
        enable_checkpointing=False
    )


    logging.info("Started training")
    start_time = time.time()  # Start timing
    trainer.fit(model)
    end_time = time.time()  # End timing
    logging.info(f"Finished training, duration: {end_time - start_time:.2f} seconds")

    logging.info("Started testing")
    start_time = time.time()  # Start timing
    trainer.test(model)
    end_time = time.time()  # End timing
    logging.info(f"Finished testing, duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_config", type=str, default="", help="Path to file that contains all the args for finetuning.", required=True)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.finetune_config, 'r') as file:
        finetune_config_dict = yaml.safe_load(file)

    # Convert the dictionary into a ConfigDict object
    finetune_config = ConfigDict(finetune_config_dict)
    run_finetune(finetune_config)