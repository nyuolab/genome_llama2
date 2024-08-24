import torch
import os
import random
import numpy as np
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from model import GenomeLlama2ForCausalLM
from util import ConfigDict
import logging
import time
import argparse
import yaml

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pretrain(pretrain_config):
    # Set the precision for matrix multiplication to medium for improved performance
    torch.set_float32_matmul_precision('medium')

    print(f"Pretrain Config: {pretrain_config}")

    # Load the tokenizer from the pretrained model specified in the configuration
    tokenizer = AutoTokenizer.from_pretrained(pretrain_config.TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the GenomeLlama2 model
    model = GenomeLlama2ForCausalLM(tokenizer, pretrain_config)


    # Set seeds for reproducibility
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(pretrain_config.SEED)
    np.random.seed(pretrain_config.SEED)
    torch.manual_seed(pretrain_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pretrain_config.SEED)

    # Define a ModelCheckpoint callback to save the model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # Monitor validation loss
        dirpath=pretrain_config.CHECKPOINT_DIR, # Directory to save checkpoints
        filename='genome_llama2-{epoch:02d}-{step}-{val_loss:.2f}', # Filename format
        save_top_k=1, # Save only the best model
        mode='min', # Minimize validation loss
    )

    # Define a TensorBoard logger for logging training progress
    logger = TensorBoardLogger(pretrain_config.LOG_DIR, name="genome_llama2_119")

    # Define a PyTorch Lightning Trainer
    trainer = Trainer(
        num_nodes=pretrain_config.NUM_NODES, # Number of nodes for distributed training
        accelerator="gpu", # Use GPUs for training
        devices=pretrain_config.GPUS, # Number of GPUs to use per node
        max_steps=pretrain_config.STEPS, # Maximum number of training steps
        accumulate_grad_batches=pretrain_config.ACCUMULATE_GRAD_BATCHES,
        precision=pretrain_config.PRECISION, # Training precision
        strategy="ddp", # Distributed Data Parallel strategy used for distributed training
        gradient_clip_val=pretrain_config.GRADIENT_CLIP_VAL, # Gradient clipping value
        callbacks=[checkpoint_callback], # List of callbacks
        enable_checkpointing=pretrain_config.ENABLE_CHECKPOINTING, # Enable checkpointing of model
        logger=logger, # Logger for logging training progress
        profiler=pretrain_config.PROFILER, # Profiler for performance monitoring
    )

    # Start training and log the duration
    logging.info("Started training")
    start_time = time.time()  # Start timing
    trainer.fit(model) # Fit the model
    end_time = time.time()  # End timing
    logging.info(f"Finished training, duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_config", type=str, default="", help="Path to file that contains all the args for pretraining.", required=True)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.pretrain_config, 'r') as file:
        pretrain_config_dict = yaml.safe_load(file)

    # Convert the dictionary into a ConfigDict object
    pretrain_config = ConfigDict(pretrain_config_dict)
    run_pretrain(pretrain_config)