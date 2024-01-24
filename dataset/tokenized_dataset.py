class Llama2GenomeTokenizedDataset:

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
        sequence_outputs = self.tokenizer(
            element['sequence'],
            truncation=True,
            padding="max_length",
            max_length=self.train_config.CONTEXT_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        attention_batch = []
        label_batch = []
        for input_ids, attention_mask in zip(sequence_outputs["input_ids"], sequence_outputs["attention_mask"]):
            input_batch.append(input_ids)
            attention_batch.append(attention_mask)

        # Tokenize labels
        for i in range(len(input_batch)):
            input_ids = input_batch[i]
            if i < len(input_batch) - 1 and input_ids[-1] != self.tokenizer.pad_token_id:
                labels = input_ids[1:] + [input_batch[i + 1][0]]
            else:
                labels = input_ids[1:] + [self.tokenizer.pad_token_id]
            
            label_batch.append(labels)
            
        return {"input_ids": input_batch, "attention_mask": attention_batch, "label_ids": label_batch}