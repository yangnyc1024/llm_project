import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)

from datetime import datetime
import pytz
import random


class PEFTModelTrainer:
    def __init__(self, model, tokenizer, train_data, eval_data, model_name="trained_model"):
        """
        Initializes the PEFTModelTrainer with model, tokenizer, train, and evaluation datasets.

        Args:
        - model: The pre-trained model to be fine-tuned.
        - tokenizer: The tokenizer corresponding to the model.
        - train_data: The dataset for training.
        - eval_data: The dataset for evaluation.
        - model_name: The directory name where the trained model will be saved.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.eval_data = eval_data
        self.model_name = model_name

    def setup_peft_config(self):
        """
        Configures the PEFT settings.

        Returns:
        - The PEFT configuration.
        """
        return LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules="all-linear", #gemma
            task_type="CAUSAL_LM"
        )
    def set_seed(self, seed = 51):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # set seed for gpu
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    def setup_training_arguments(self):
        """
        Configures the training arguments.

        Returns:
        - The TrainingArguments configuration.
        """
        return TrainingArguments(
            output_dir="logs",
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            optim="paged_adamw_32bit",
            save_steps=0,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="tensorboard",
            evaluation_strategy="epoch",
            seed = 51
        )
    def train_model(self):
        """
        Executes the training process using PEFT.
        """
        peft_config = self.setup_peft_config()
        self.set_seed()
        training_arguments = self.setup_training_arguments()


        # Initialize the custom PEFT Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
            max_seq_length=1024,
        )

        # Training process
        tz = pytz.timezone("America/New_York")
        start_time = datetime.now(tz)
        print(f'\nTraining started at {start_time}')

        trainer.train()

        end_time = datetime.now(tz)
        duration = end_time - start_time
        print(f'Training completed at {end_time}')
        print(f'Training duration was {duration}')

        # Save the trained model
        trainer.model.save_pretrained(self.model_name)

# # Assuming 'model', 'tokenizer', 'train_data', and 'eval_data' have been previously defined and initialized
# trainer = PEFTModelTrainer(model, tokenizer, train_data, eval_data)

# # Start the training process
# trainer.train_model()
