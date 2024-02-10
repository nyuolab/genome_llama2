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

def main():

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
        filename='llama2_genome_7b-{val_loss:.2f}',
        # filename='llama2_genome_13b-{val_loss:.2f}',     # Uncomment for training Llama2 13b model
        save_top_k=1,
        mode='min',
    )

    # Logger
    logger = TensorBoardLogger("llama2_genome_tb_logs", name="llama2_genome_7b")
    # logger = TensorBoardLogger("llama2_genome_tb_logs", name="llama2_genome_13b")    # Uncomment for training Llama2 13b model

    # Define lightning trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=train_config.GPUS,
        max_epochs=train_config.EPOCHS,
        precision=train_config.PRECISION,
        strategy="deepspeed_stage_2_offload",
        # strategy="deepspeed_stage_3_offload",    # Uncomment for training Llama2 13b model
        accumulate_grad_batches=train_config.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=train_config.GRADIENT_CLIP_VAL,
        callbacks=[checkpoint_callback],
        enable_checkpointing=train_config.ENABLE_CHECKPOINTING,
        logger=logger,
        profiler=train_config.PROFILER
    )

    print("Started training")
    trainer.fit(llama2_genome)
    print("Finished training")

if __name__ == "__main__":
    main()