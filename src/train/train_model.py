from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelWithLMHead
from transformers import TextDataset,DataCollatorForLanguageModeling

import torch
import logging
import os

import src.generate.generate_training_data as training_data

logging.info(f"cuda is: {torch.cuda.is_available()}")

MODEL_NAME = "anonymous-german-nlp/german-gpt2"
TRAINED_MODEL_PATH = f"{os.getcwd()}/gpt2-discord_chat_merged"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelWithLMHead.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token # fix padding issue with gpt-2

def run():
    train_dataset, test_dataset, data_collator = _load_dataset(training_data.TRAINING_DATA_PATH, training_data.VALIDATION_DATA_PATH, tokenizer)

    training_args = TrainingArguments(
        output_dir=TRAINED_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_steps = 400,
        save_steps=800,
        warmup_steps=500,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    trainer.save_model()

def _load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator

if __name__ == "__main__":
    raise SystemExit(run())