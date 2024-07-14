import torch
from transformers import AutoConfig, LlamaForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk
from dataset.genome_dataset import GenomeDataset
from lightning.pytorch import LightningModule
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

# Define a LightningModule for training a Llama-2 model with genome data
class GenomeLlama2(LightningModule):
    def __init__(self, tokenizer, model_config):
        super().__init__()
        self.model = None
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.genome_train = None
        self.genome_valid = None
        self.train_sampler = None
        self.valid_sampler = None
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up the datasets and samplers
    def setup(self, stage=None):
        # Assuming the dataset is already prepared and saved on disk
        if stage == "fit" or stage is None:
            tokenized_genome_dataset = load_from_disk(self.model_config.TOKENIZED_DATASET_PATH)
            self.genome_train = GenomeDataset(tokenized_genome_dataset["train"])
            self.genome_valid = GenomeDataset(tokenized_genome_dataset["valid"])
            self.train_sampler = DistributedSampler(self.genome_train, shuffle=True)
            self.valid_sampler = DistributedSampler(self.genome_valid, shuffle=False)

    # Forward pass of the model
    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    # Helper function to compute the loss for a batch
    def _step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        return loss

    # Training step
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True) # Log the training loss
        return loss

    # Validation step
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True) # Log the validation loss
        return loss
    
    # Configure the model with specific parameters to match the architecture of DNABERT-2
    def configure_model(self):
        if self.model is not None:
            return
        
        # Load the base configuration for the Llama model from pretrained settings
        config = AutoConfig.from_pretrained(
            self.model_config.LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=self.model_config.CONTEXT_LENGTH
        )

        # Set the model parameters to align with DNABERT-2 architecture
        config.num_hidden_layers = 12 # Number of transformer layers
        config.num_attention_heads = 12 # Number of attention heads
        config.num_key_value_heads = 12 # Number of key-value heads in attention mechanism
        config.hidden_size = 768 # Size of hidden layers
        config.intermediate_size = 3072 # Size of intermediate layers in the feed-forward network

        # Set the beginning of sequence (bos) and end of sequence (eos) token IDs
        config.bos_token_id=self.tokenizer.bos_token_id
        config.eos_token_id=self.tokenizer.eos_token_id

        # Initialize the model with the configured settings
        self.model = LlamaForCausalLM(config)

    # Configure the optimizer and learning rate scheduler
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr=self.model_config.LR_MAX, 
                                     betas=self.model_config.BETAS, 
                                     eps=self.model_config.EPS, 
                                     weight_decay=self.model_config.WEIGHT_DECAY)

        scheduler = OneCycleLR(
            optimizer,
            max_lr = self.model_config.LR_MAX, # Upper learning rate boundaries in the cycle for each parameter group
            total_steps = self.model_config.STEPS, # Total number of steps to train for.
            anneal_strategy = 'linear',
            pct_start = float(self.model_config.WARM_UP/self.model_config.STEPS))

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]
    
    # Create a DataLoader for the training set
    def train_dataloader(self):
        return DataLoader(self.genome_train, batch_size=self.model_config.BATCH_SIZE, num_workers = self.model_config.NUM_WORKERS, sampler=self.train_sampler, shuffle=(self.train_sampler is None), pin_memory=True, collate_fn=self.data_collator)

    # Create a DataLoader for the validation set
    def val_dataloader(self):
        return DataLoader(self.genome_valid, batch_size=self.model_config.BATCH_SIZE, sampler=self.valid_sampler, num_workers = self.model_config.NUM_WORKERS, shuffle=False, pin_memory=True, collate_fn=self.data_collator)