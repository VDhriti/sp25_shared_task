from datetime import datetime
import argparse

def time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------- argument parsing ----------
parser = argparse.ArgumentParser(
    description="Fine-tune BERTweet on one of the shared-task datasets."
)
parser.add_argument(            # ① add a *positional* argument
    "dataset",                  #    (makes the flag mandatory)
    choices=["easy", "medium", "hard"],
    help="Which dataset split to train on"
)
# If you’d rather have an *optional* flag, use:
# parser.add_argument("--dataset", default="medium", choices=[...])

args = parser.parse_args()

DATASET = args.dataset          # ② replace the hard-coded string
SAVE_FOLDER = f"bertweet_large_cr_{DATASET}_{time()}"

print(time(), f"Training on the {DATASET} dataset")



import numpy as np
import json
import os
from glob import glob
from typing import List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import Dataset

# METRICS
def compute_metrics(eval_pred):
    """
    Hugging-Face Trainer passes a named-tuple (logits, labels).

    Returns:
        dict with metric-name → float
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    # average='binary' because it’s a 0/1 task; use 'macro' for multi-class
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
    }

# DATA
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir: str, tokenizer, max_length: int = 512):
        """
        root_dir should be e.g. "easy/train" or "hard/validation"
        Expects files: problem-*.txt and truth-problem-*.json
        """
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # find all txt files
        for txt_path in glob(os.path.join(root_dir, "problem-*.txt")):
            base = os.path.splitext(os.path.basename(txt_path))[0]  # e.g. "problem-3"
            json_path = os.path.join(root_dir, f"truth-{base}.json")
            if not os.path.exists(json_path):
                continue

            # read sentences
            with open(txt_path, encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            # read labels
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            changes: List[int] = data["changes"]

            # build pairs (sent_i, sent_{i+1})
            for i, label in enumerate(changes):
                if i + 1 < len(lines):
                    self.examples.append({
                        "sent1": lines[i],
                        "sent2": lines[i+1],
                        "label": label,
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # tokenizer will handle truncation and padding (padding done in collator)
        enc = self.tokenizer(
            ex["sent1"],
            ex["sent2"],
            truncation=True,
            max_length=self.max_length,
            #padding="max_length",
        )
        enc["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return enc
    
#LOAD MODEL, TOKENIZER
print(time(), "LOADING MODEL, TOKENIZER")

bertweet = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", num_labels=2, 
                                                              problem_type="single_label_classification",
                                                              id2label={0: "NEG", 1: "POS"}, label2id={"NEG": 0, "POS": 1},
                                                             )
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", use_fast=True, 
                                          normalization=True, add_prefix_space=True,)

train_ds = ChangeDetectionDataset(f"{DATASET}/train", tokenizer)
eval_ds  = ChangeDetectionDataset(f"{DATASET}/validation", tokenizer)
print(time(), "DATASETS LOADED")


# DATA COLLATOR (cause of issues)
data_collator = DataCollatorWithPadding(
    tokenizer,                # uses the model’s pad-token ID
    pad_to_multiple_of=8,     # keeps tensors tensor-core friendly; optional
    return_tensors="pt"       # default, explicit for clarity
)

# TRAINING ARGS
training_args = TrainingArguments(
    output_dir=SAVE_FOLDER,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,      # ↓ if you OOM; ↑ if you have VRAM
    auto_find_batch_size = True,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,                          # half-precision = 2× speed, 2× memory
    gradient_accumulation_steps=2,      # acts like batch_size double
    logging_steps=50,
    # extra optimization
    # dataloader_num_workers=4,
)

trainer = Trainer(
    model=bertweet,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,   # keep this so metrics decode nicely
    data_collator=data_collator,   # <-- perfectly OK to omit
    compute_metrics = compute_metrics,
)

# TRAINING
print(time(), "BEGINNING TRAINING")
trainer.train()

print(time(), "ENDING TRAINING")

metrics = trainer.evaluate()
print(metrics)

trainer.save_model(f"{SAVE_FOLDER}/final")
tokenizer.save_pretrained(f"{SAVE_FOLDER}/final")
print(time(), "MODELS SAVED")
