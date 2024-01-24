import torch
import os
import random
import numpy as np
import pytorch_lightning as pl
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from datasets import DatasetDict
from transformers import AutoTokenizer
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data_processing.fasta_processor import FastaGenomeSequenceProcessor
from tokenization.tokenizer_trainer import Llama2GenomeTokenizerTrainer
from dataset.tokenized_dataset import Llama2GenomeTokenizedDataset
from model.llama2_genome import Llama2Genome
from config import TrainConfig

def main():

    train_config = TrainConfig()

    # Preprocess Genome Data
    preprocessor = FastaGenomeSequenceProcessor(train_config.RAW_DATA_FILE_PATH, train_config)
    sequences = preprocessor.process_fasta_file()
    train_size = int(len(sequences) * train_config.TRAIN_RATIO)
    val_size = int(len(sequences) * train_config.VALID_RATIO)
    ds_train, ds_valid, ds_test = preprocessor.split_dataset(sequences, train_size, val_size)
    raw_datasets = DatasetDict({
        "train": preprocessor.create_dataset(ds_train),
        "valid": preprocessor.create_dataset(ds_valid),
        "test": preprocessor.create_dataset(ds_test)
    })


    # Tokenizer training
    trainer = Llama2GenomeTokenizerTrainer(raw_datasets['train'], train_config)
    tokenizer = trainer.train_new_tokenizer()
    print("Tokenizer trained successfully.")
    tokenizer.save_pretrained(train_config.TOKENIZER_SAVE_PATH)
    print(f"Tokenizer saved successfully to {train_config.TOKENIZER_SAVE_PATH}.")

    # Tokenizer usage example
    tokenizer = AutoTokenizer.from_pretrained(train_config.TOKENIZER_SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    example = "AGCTTAGCTAGTCGTAGCTAATCGATCGATCGATCGTAGCTAGCTAGCTAAGCTTAGCTA"
    tokens = tokenizer.tokenize(example)
    print(tokens)


    # Tokenized dataset
    tokenizedDataset = Llama2GenomeTokenizedDataset(tokenizer, train_config)
    tokenizedDataset = tokenizedDataset.shared_transform(raw_datasets)


    llama2_genome = Llama2Genome(tokenizer, tokenizedDataset, train_config.LR, train_config)


    # Initialize with a seed
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # Auto Wrap Policy
    policy = partial(size_based_auto_wrap_policy, min_num_params=1000000)

    # ModelCheckpoint Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Replace 'val_loss' with your validation metric
        dirpath='llama2_genome_checkpoint',  # Path where the checkpoints will be saved
        filename='llama2_genome-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Number of best models to save; set to -1 to save all checkpoints
        mode='min',  # 'min' for metrics where lower is better (like loss), 'max' for accuracy
    )

    # EarlyStopping Callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # or another metric
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'
    )

    # Logger
    logger = TensorBoardLogger("llama2_genome_tb_logs", name="llama2_genome")

    # FSDP Strategy
    strategy = FSDPStrategy(
        sharding_strategy='FULL_SHARD',
        auto_wrap_policy=policy,
        limit_all_gathers=True,
        cpu_offload=True,
        mixed_precision='mixed',  # Ensure mixed precision is enabled
        )

    # Define lightning trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=train_config.GPUS,
        max_epochs=train_config.EPOCHS,
        precision=train_config.PRECISION,
        strategy=strategy,
        accumulate_grad_batches=train_config.ACCUMULATE_GRAD_BATCHES,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=train_config.ENABLE_CHECKPOINTING,
        logger=logger,
        profiler=train_config.PROFILER
        )

    print("Started training")
    trainer.fit(llama2_genome)
    print("Finished training")

if __name__ == "__main__":
    main()