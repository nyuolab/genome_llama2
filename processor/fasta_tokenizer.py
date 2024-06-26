from datasets import DatasetDict
from transformers import AutoTokenizer
from config import TrainConfig
from processor.fasta_processor import GenomeSequenceProcessor

class GenomeSequenceTokenizer():

    def __init__(self, tokenizer, train_config):
        self.tokenizer = tokenizer
        self.train_config = train_config

    def shared_transform(self, processed_dataset):
        tokenized_ds = processed_dataset.map(
            self.tokenize,
            remove_columns=processed_dataset["train"].column_names,
            batched=True,
            load_from_cache_file=True,
        )

        return tokenized_ds

    def tokenize(self, element):
        outputs = self.tokenizer(
            element['sequence'],
            truncation=True,
            padding="max_length",
            max_length=self.train_config.CONTEXT_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
            return_tensors="pt"
        )
        
        labels = outputs["input_ids"].clone()
        shifted_labels = labels[:, 1:].clone()  # Clone the shifted tensor
        labels[:, :-1] = shifted_labels
        labels[:, -1] = self.tokenizer.pad_token_id
            
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "labels": labels}
    
def tokenize_dataset():
    train_config = TrainConfig()

    tokenizer = AutoTokenizer.from_pretrained(train_config.TOKENIZER_SAVE_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    preprocessor = GenomeSequenceProcessor(train_config.RAW_DATA_DIR_PATH, train_config)
    ds_train, ds_valid = preprocessor.create_and_split_dataset(train_config.TRAIN_RATIO, chunk_size=10000)
    raw_datasets = DatasetDict({
        "train": ds_train,
        "valid": ds_valid
    })

    tokenizer = GenomeSequenceTokenizer(tokenizer, train_config)

    print("Genome dataset tokenization started...")
    tokenized_dataset = tokenizer.shared_transform(raw_datasets)
    print("Genome dataset tokenized successfully...")
    tokenized_dataset.save_to_disk(train_config.TOKENIZED_DATASET_SAVE_PATH)
    print(f"Tokenized Genome dataset saved successfully to {train_config.TOKENIZED_DATASET_SAVE_PATH}.")

if __name__ == "__main__":
    tokenize_dataset()