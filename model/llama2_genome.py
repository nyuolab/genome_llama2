from transformers import AutoConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_from_disk
from dataset.genome_dataset import GenomeDataset
from lightning.pytorch import LightningModule
import math
from torch.utils.data.distributed import DistributedSampler
from deepspeed.ops.adam import FusedAdam
from torch.optim.lr_scheduler import OneCycleLR

class Llama2Genome(LightningModule):
    def __init__(self, tokenizer, train_config):
        super().__init__()
        self.model = None
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.genome_train = None
        self.genome_valid = None
        self.train_sampler = None
        self.valid_sampler = None

    def setup(self, stage=None):
        # Assuming the dataset is already prepared and saved on disk
        if stage == "fit" or stage is None:
            tokenized_genome_dataset = load_from_disk(self.train_config.TOKENIZED_DATASET_SAVE_PATH)
            self.genome_train = GenomeDataset(tokenized_genome_dataset["train"])
            self.genome_valid = GenomeDataset(tokenized_genome_dataset["valid"])
            self.train_sampler = DistributedSampler(self.genome_train, shuffle=True)
            self.valid_sampler = DistributedSampler(self.genome_valid, shuffle=False)
            self.effective_batch_size = self.train_config.BATCH_SIZE * self.train_config.ACCUMULATE_GRAD_BATCHES * self.train_config.NUM_NODES * self.train_config.GPUS
            self.steps_per_epoch = int(len(self.genome_train) / self.effective_batch_size)
            self.total_steps = self.steps_per_epoch * self.train_config.EPOCHS

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def _step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, _ = self(input_ids, attention_mask, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_model(self):
        if self.model is not None:
            return
        config = AutoConfig.from_pretrained(
            self.train_config.LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=self.train_config.CONTEXT_LENGTH,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model = LlamaForCausalLM(config)

    def configure_optimizers(self):
        optimizer = FusedAdam(self.parameters(), 
                                     lr=self.train_config.LR_MAX, 
                                     betas=self.train_config.BETAS, 
                                     eps=self.train_config.EPS, 
                                     weight_decay=self.train_config.WEIGHT_DECAY)

        scheduler = OneCycleLR(
            optimizer,
            max_lr = self.train_config.LR_MAX, # Upper learning rate boundaries in the cycle for each parameter group
            total_steps = self.trainer.estimated_stepping_batches, # The number of steps per epoch to train for.
            epochs = self.train_config.EPOCHS, # The number of epochs to train for.
            anneal_strategy = 'cos',
            pct_start = float(self.train_config.WARM_UP/self.trainer.estimated_stepping_batches))

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        return DataLoader(self.genome_train, batch_size=self.train_config.BATCH_SIZE, num_workers = self.train_config.NUM_WORKERS, sampler=self.train_sampler, shuffle=(self.train_sampler is None), pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.genome_valid, batch_size=32, sampler=self.valid_sampler, num_workers = self.train_config.NUM_WORKERS, shuffle=False, pin_memory=True)