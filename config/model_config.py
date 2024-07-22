from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Constants related to the DNA sequences and model
    CONTEXT_LENGTH: int = 128
    LLM_MODEL: str = "meta-llama/Llama-2-7b-hf"
    NEW_TOKENIZER_VOCAB_SIZE: int = 4094

    # File paths
    RAW_TRAIN_DATA_DIR_PATH: str = "dnabert_2_pretrain"
    TOKENIZER_PATH: str = "genome_llama2/tokenizer"
    TOKENIZED_DATASET_PATH: str = "genome_tokenized_dataset"
    CHECKPOINT_DIR: str = "genome_llama2_checkpoint"
    LOG_DIR: str = "genome_llama2_tb_logs"

    # Model Type
    MODEL_SIZE: str = "base"

    # Training arguments
    STEPS: int = 500000
    NUM_NODES: int = 1
    GPUS: int = -1
    LR_MAX: float = 5e-4
    WARM_UP: int = 30000
    BETAS: tuple = (0.9, 0.98)
    EPS: float = 1e-6
    WEIGHT_DECAY: float = 1e-5
    PRECISION: str = "bf16-mixed"
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 0
    ACCUMULATE_GRAD_BATCHES: int = 2
    GRADIENT_CLIP_VAL: float = 1.0
    ENABLE_CHECKPOINTING: bool = True
    PROFILER: str = 'advanced'

    # Set seed for reproducibility
    SEED: int = 1234