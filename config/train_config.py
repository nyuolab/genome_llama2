from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Constants related to the DNA sequences and model
    CONTEXT_LENGTH: int = 1024
    LLM_MODEL: str = "meta-llama/Llama-2-7b-hf"
    NEW_TOKENIZER_VOCAB_SIZE: int = 4096

    # File paths
    RAW_DATA_DIR_PATH: str = "genome_llama2/genome_train_model_raw_data"
    TOKENIZER_SAVE_PATH: str = "llama2_genome_tokenizer"
    TOKENIZED_DATASET_SAVE_PATH: str = "genome_tokenized_dataset"
    CHECKPOINT_DIR: str = "llama2_genome_checkpoint"
    LOG_DIR: str = "llama2_genome_tb_logs"

    # Dataset split ratios
    TRAIN_RATIO: float = 0.9
    # VALID_RATIO set to 0.1.

    # Training arguments
    EPOCHS: int = 2
    NUM_NODES: int = 2
    GPUS: int = -1
    LR_MAX: float = 3e-4
    WARM_UP: int = 16000
    BETAS: tuple = (0.9, 0.999)
    EPS: float = 1e-8
    WEIGHT_DECAY: float = 1e-5
    PRECISION: str = "bf16-mixed"
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 0
    ACCUMULATE_GRAD_BATCHES: int = 128
    GRADIENT_CLIP_VAL: float = 1.0
    ENABLE_CHECKPOINTING: bool = True
    PROFILER: str = 'advanced'

    # Set seed for reproducibility
    SEED: int = 1234