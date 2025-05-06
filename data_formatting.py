from typing import List, Dict, Tuple
import os, json
from glob import glob

import torch
from torch.utils.data import Dataset


class TripletChangeDetectionDataset(Dataset):
    """
    Example  = (sent_i, sent_{i+1})   ⟂   sent_{i+2}
    Label    = changes[i+1]           (author‑change flag after the second sentence)
    """

    def __init__(self, root_dir: str, tokenizer, max_length: int = 512, verbose: bool = True):
        super().__init__()
        self.examples: List[Dict[str, str | int]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # bookkeeping for skipped files
        self.skipped_files: List[str] = []

        # -------------------------------------------------- #
        # iterate through all problem‑*.txt files
        # -------------------------------------------------- #
        for txt_path in glob(os.path.join(root_dir, "problem-*.txt")):
            base = os.path.splitext(os.path.basename(txt_path))[0]      # e.g. "problem‑1234"
            json_path = os.path.join(root_dir, f"truth-{base}.json")
            if not os.path.exists(json_path):
                self.skipped_files.append(f"{base} (missing truth file)")
                continue

            with open(txt_path, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            with open(json_path, encoding="utf-8") as f:
                changes: List[int] = json.load(f)["changes"]

            # skip malformed file outright
            if len(changes) != len(lines) - 1:
                self.skipped_files.append(f"{base} (len(lines)={len(lines)}, len(changes)={len(changes)})")
                continue

            # build sliding‑window triplets
            for i in range(len(lines) - 2):
                self.examples.append(
                    {
                        "context_pair": (lines[i], lines[i + 1]),
                        "candidate":     lines[i + 2],
                        "label":         changes[i + 1],
                    }
                )

        # ---------- optional summary ----------
        if verbose:
            kept   = len(set(os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(root_dir, "problem-*.txt")))) - len(self.skipped_files)
            skipped = len(self.skipped_files)
            print(f"[TripletDataset] loaded {kept} files, skipped {skipped}")

    # --------------------------------------------------------------------------- #
    # PyTorch dataset protocol
    # --------------------------------------------------------------------------- #
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        sep_tok = self.tokenizer.sep_token            # "</s>" for RoBERTa/BERTweet
        text_a  = f"{ex['context_pair'][0]} {sep_tok} {ex['context_pair'][1]}"
        text_b  = ex["candidate"]

        enc = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return enc


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