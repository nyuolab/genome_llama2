import torch
import os
import random
import numpy as np
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from model.llama2_genome import Llama2Genome
from config import TrainConfig
from lightning.pytorch.strategies import DeepSpeedStrategy
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():

    torch.set_float32_matmul_precision('medium')

    train_config = TrainConfig()

    tokenizer = AutoTokenizer.from_pretrained(train_config.TOKENIZER_SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    llama2_genome = Llama2Genome(tokenizer, train_config)


    # Initialize with a seed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(train_config.SEED)
    np.random.seed(train_config.SEED)
    torch.manual_seed(train_config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.SEED)

    # ModelCheckpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='llama2_genome_checkpoint',
        filename='llama2_genome_7b-{epoch:02d}-{step}-{val_loss:.2f}',
        save_top_k=1,
        every_n_train_steps=103500,
        mode='min',
    )

    # Logger
    logger = TensorBoardLogger("llama2_genome_tb_logs", name="llama2_genome_7b")

    # Define lightning trainer
    trainer = Trainer(
        num_nodes=train_config.NUM_NODES,
        accelerator="gpu",
        devices=train_config.GPUS,
        max_epochs=train_config.EPOCHS,
        precision=train_config.PRECISION,
        strategy=DeepSpeedStrategy(
            stage=3,
            logging_level=logging.DEBUG,
            zero_optimization=True,
            allgather_partitions=True, 
            reduce_scatter=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            overlap_comm=True,
            contiguous_memory_optimization=False,
            synchronize_checkpoint_boundary=True
        ),
        gradient_clip_val=train_config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback],
        enable_checkpointing=train_config.ENABLE_CHECKPOINTING,
        logger=logger,
        profiler=train_config.PROFILER
    )

    logging.info("Started training")
    start_time = time.time()  # Start timing
    trainer.fit(llama2_genome)
    end_time = time.time()  # End timing
    logging.info(f"Finished training, duration: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()