from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Constants related to the DNA sequences and model
    MIN_SEQUENCE_LENGTH: int = 1000
    CONTEXT_LENGTH: int = 1024
    LLM_MODEL: str = "meta-llama/Llama-2-7b-hf"
    # LLM_MODEL: str = "meta-llama/Llama-2-13b-hf"  # Uncomment for training Llama2 13b model
    NEW_TOKENIZER_VOCAB_SIZE: int = 4096

    # File paths
    RAW_DATA_FILE_PATH: str = "genome_data/GRCh38_genomic.txt"
    TOKENIZER_SAVE_PATH: str = "llama2_genome_tokenizer"
    TOKENIZED_DATASET_SAVE_PATH: str = "genome_tokenized_dataset"
    CHECKPOINT_DIR: str = "llama2_genome_checkpoint"
    LOG_DIR: str = "llama2_genome_tb_logs"

    # Dataset split ratios
    TRAIN_RATIO: float = 0.8
    # VALID_RATIO and TEST_RATIO set to 0.5 * (1 - TRAIN_RATIO) each by default.

    # Training arguments
    EPOCHS: int = 1
    GPUS: int = -1
    LR: float = 3e-4
    BETAS: tuple = (0.9, 0.95)
    EPS: float = 1e-5
    WEIGHT_DECAY: float = 0.1
    WARMUP: int = 2000
    PRECISION: str = "bf16"
    BATCH_SIZE: int = 2
    # BATCH_SIZE: int = 1    # Uncomment for training Llama2 13b model
    NUM_WORKERS: int = 0
    ACCUMULATE_GRAD_BATCHES: int = 8
    # ACCUMULATE_GRAD_BATCHES: int = 16    # Uncomment for training Llama2 13b model
    GRADIENT_CLIP_VAL: float = 1.0
    ENABLE_CHECKPOINTING: bool = True
    PROFILER: str = 'advanced'

    # Set seed for reproducibility
    SEED: int = 1234