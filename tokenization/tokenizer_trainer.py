from transformers import AutoTokenizer

class Llama2GenomeTokenizerTrainer:

    def __init__(self, dataset, train_config):
        self.dataset = dataset
        self.train_config = train_config

    def get_training_corpus(self, chunk_size=1000):
        return (self.dataset["sequence"][i : i + chunk_size] for i in range(0, len(self.dataset), chunk_size))

    def train_tokenizer(self):
        training_corpus = self.get_training_corpus()
        old_tokenizer = AutoTokenizer.from_pretrained(self.train_config.LLM_MODEL)
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, self.train_config.NEW_TOKENIZER_VOCAB_SIZE)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
        return tokenizer