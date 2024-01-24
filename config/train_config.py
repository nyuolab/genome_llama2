from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Constants related to the DNA sequences and model
    MIN_SEQUENCE_LENGTH: int = 100000
    CONTEXT_LENGTH: int = 128
    LLM_MODEL: str = "meta-llama/Llama-2-7b-hf"
    NEW_TOKENIZER_VOCAB_SIZE: int = 1000

    # File paths
    RAW_DATA_FILE_PATH: str = "genome_data/GRCh38_genomic.txt"
    TOKENIZER_SAVE_PATH: str = "llama2_genome_tokenizer"
    CHECKPOINT_DIR: str = "llama2_genome_checkpoint"
    LOG_DIR: str = "llama2_genome_tb_logs"

    # Dataset split ratios
    TRAIN_RATIO: float = 0.8
    VALID_RATIO: float = 0.9

    # Training arguments
    EPOCHS: int = 20
    GPUS: int = -1
    LR: float = 1e-3
    PRECISION: int = 16
    BATCH_SIZE: int = 8
    NUM_WORKERS: int = 4
    ACCUMULATE_GRAD_BATCHES: int = 2
    ENABLE_CHECKPOINTING: bool = False
    PROFILER: str = 'advanced'

    # Set seed for reproducibility
    SEED: int = 1234