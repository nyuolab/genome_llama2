import torch
from transformers import AutoConfig, LlamaForCausalLM, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk
from dataset.genome_dataset import GenomeDataset
from lightning.pytorch import LightningModule
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import OneCycleLR

class GenomeLlama2ForCausalLM(LightningModule):
    """A LightningModule for pretraining a Llama-2 Causal LM with genome data"""

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

    def setup(self, stage=None):
        """Set up the datasets and samplers"""

        # Assuming the dataset is already prepared and saved on disk
        if stage == "fit" or stage is None:
            tokenized_genome_dataset = load_from_disk(self.model_config.TOKENIZED_DATASET_PATH)
            self.genome_train = GenomeDataset(tokenized_genome_dataset["train"])
            self.genome_valid = GenomeDataset(tokenized_genome_dataset["valid"])
            self.train_sampler = DistributedSampler(self.genome_train, shuffle=True)
            self.valid_sampler = DistributedSampler(self.genome_valid, shuffle=False)

    def forward(self, input_ids, attention_mask, labels):
        """Forward pass of the model"""
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def _step(self, batch):
        """Helper function to compute the loss for a batch"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True) # Log the training loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True) # Log the validation loss
        return loss
    
    def configure_model(self):
        """Configure the model with specific parameters to match the architecture of DNABERT-2"""
        if self.model is not None:
            return
        
        # Load the base configuration for the Llama model from pretrained settings
        config = AutoConfig.from_pretrained(
            self.model_config.LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=self.model_config.CONTEXT_LENGTH
        )

        # Set the model parameters
        # Total number of parameters: 119 M
        if(self.model_config.MODEL_SIZE == "base"):
            # The following model parameters align with DNABERT-2 architecture
            config.num_hidden_layers = 12 # Number of transformer layers
            config.num_attention_heads = 12 # Number of attention heads
            config.num_key_value_heads = 12 # Number of key-value heads in attention mechanism
            config.hidden_size = 768 # Size of hidden layers
            config.intermediate_size = 3072 # Size of intermediate layers in the feed-forward network

        # Total number of parameters: 411 M
        elif(self.model_config.MODEL_SIZE == "medium"):
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.num_key_value_heads = 16
            config.hidden_size = 1024
            config.intermediate_size = 4096

        # Total number of parameters: 744 M
        elif(self.model_config.MODEL_SIZE == "large"):
            config.num_hidden_layers = 28
            config.num_attention_heads = 20
            config.num_key_value_heads = 20
            config.hidden_size = 1280
            config.intermediate_size = 5120

        # Set the beginning of sequence (bos) and end of sequence (eos) token IDs
        config.bos_token_id=self.tokenizer.bos_token_id
        config.eos_token_id=self.tokenizer.eos_token_id

        # Initialize the model with the configured settings
        self.model = LlamaForCausalLM(config)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler"""

        # Explicitly cast config values to float and int
        lr_max = float(self.model_config.LR_MAX)
        betas = tuple(map(float, self.model_config.BETAS))
        eps = float(self.model_config.EPS)
        weight_decay = float(self.model_config.WEIGHT_DECAY)
        warm_up = float(self.model_config.WARM_UP)
        steps = int(self.model_config.STEPS)


        optimizer = torch.optim.AdamW(self.parameters(), 
                                     lr=lr_max, 
                                     betas=betas, 
                                     eps=eps, 
                                     weight_decay=weight_decay)

        scheduler = OneCycleLR(
            optimizer,
            max_lr = lr_max, # Upper learning rate boundaries in the cycle for each parameter group
            total_steps = steps, # Total number of steps to train for.
            anneal_strategy = 'linear',
            pct_start = float(warm_up/steps))

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        """Create a DataLoader for the training set"""
        return DataLoader(self.genome_train, batch_size=self.model_config.BATCH_SIZE, num_workers = self.model_config.NUM_WORKERS, sampler=self.train_sampler, shuffle=(self.train_sampler is None), pin_memory=True, collate_fn=self.data_collator)

    def val_dataloader(self):
        """Create a DataLoader for the validation set"""
        return DataLoader(self.genome_valid, batch_size=self.model_config.BATCH_SIZE, sampler=self.valid_sampler, num_workers = self.model_config.NUM_WORKERS, shuffle=False, pin_memory=True, collate_fn=self.data_collator)