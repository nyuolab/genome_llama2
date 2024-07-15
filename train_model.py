import torch
import os
import random
import numpy as np
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from model import GenomeLlama2
from config import ModelConfig
import logging
import time
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run(model_config):
    # Set the precision for matrix multiplication to medium for improved performance
    torch.set_float32_matmul_precision('medium')

    # Load the tokenizer from the pretrained model specified in the configuration
    tokenizer = AutoTokenizer.from_pretrained(model_config.TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the GenomeLlama2 model
    llama2_genome = GenomeLlama2(tokenizer, model_config)


    # Set seeds for reproducibility
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(model_config.SEED)
    np.random.seed(model_config.SEED)
    torch.manual_seed(model_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_config.SEED)

    # Define a ModelCheckpoint callback to save the model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # Monitor validation loss
        dirpath=model_config.CHECKPOINT_DIR, # Directory to save checkpoints
        filename='genome_llama2-{epoch:02d}-{step}-{val_loss:.2f}', # Filename format
        save_top_k=1, # Save only the best model
        mode='min', # Minimize validation loss
    )

    # Define a TensorBoard logger for logging training progress
    logger = TensorBoardLogger(model_config.LOG_DIR, name="genome_llama2_119")

    # Define a PyTorch Lightning Trainer
    trainer = Trainer(
        num_nodes=model_config.NUM_NODES, # Number of nodes for distributed training
        accelerator="gpu", # Use GPUs for training
        devices=model_config.GPUS, # Number of GPUs to use per node
        max_steps=model_config.STEPS, # Maximum number of training steps
        precision=model_config.PRECISION, # Training precision
        strategy="ddp", # Distributed Data Parallel strategy used for distributed training
        gradient_clip_val=model_config.GRADIENT_CLIP_VAL, # Gradient clipping value
        callbacks=[checkpoint_callback], # List of callbacks
        enable_checkpointing=model_config.ENABLE_CHECKPOINTING, # Enable checkpointing of model
        logger=logger, # Logger for logging training progress
        profiler=model_config.PROFILER, # Profiler for performance monitoring
    )

    # Start training and log the duration
    logging.info("Started training")
    start_time = time.time()  # Start timing
    trainer.fit(llama2_genome) # Fit the model
    end_time = time.time()  # End timing
    logging.info(f"Finished training, duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_dataset_path", type=str, default="", help="Path to the tokenized dataset directory.", required=True)
    parser.add_argument("--checkpoint_dir_path", type=str, default="", help="Path to the directory where model checkpoints will be saved during training.", required=True)
    parser.add_argument("--log_dir_path", type=str, default="", help="Path to the directory where training logs and other related outputs will be saved.", required=True)
    parser.add_argument("--n_nodes", type=int, default=1, help="Number of nodes to be used in the training process.")
    parser.add_argument("--n_gpus", type=int, default=-1, help="Number of GPUs to be used per node. Use -1 to utilize all available GPUs.")
    args = parser.parse_args()

    model_config = ModelConfig()
    model_config.TOKENIZED_DATASET_PATH = args.tokenized_dataset_path
    model_config.CHECKPOINT_DIR = args.checkpoint_dir_path
    model_config.LOG_DIR = args.log_dir_path
    model_config.NUM_NODES = args.n_nodes
    model_config.GPUS = args.n_gpus

    run(model_config)