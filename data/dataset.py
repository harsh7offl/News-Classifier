"""
Dataset loading and preprocessing for AG News classification.
Dataset: AG News (4 classes: World, Sports, Business, Sci/Tech)
Source: HuggingFace datasets
"""

from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd


class AGNewsDataset(Dataset):
    """Custom PyTorch Dataset wrapper for AG News."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_agnews(split: str = "train", max_samples: int = None):
    """
    Load AG News dataset from HuggingFace.

    Args:
        split: 'train' or 'test'
        max_samples: limit samples for quick experimentation

    Returns:
        texts (list), labels (list)
    """
    dataset = load_dataset("fancyzhx/ag_news", split=split)

    if max_samples:
        dataset = dataset.select(range(max_samples))

    texts = [item["text"] for item in dataset]
    labels = [item["label"] for item in dataset]

    return texts, labels


def get_label_map():
    """Return AG News label index to class name mapping."""
    return {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech"
    }


def preprocess_text(text: str) -> str:
    """
    Basic text cleaning.

    Args:
        text: raw input string

    Returns:
        cleaned string
    """
    text = text.strip()
    text = " ".join(text.split())  # normalize whitespace
    return text


if __name__ == "__main__":
    print("Loading AG News dataset...")
    texts, labels = load_agnews(split="train", max_samples=1000)
    label_map = get_label_map()

    df = pd.DataFrame({"text": texts, "label": labels})
    df["label_name"] = df["label"].map(label_map)

    print(f"\nLoaded {len(df)} samples")
    print("\nClass distribution:")
    print(df["label_name"].value_counts())
    print("\nSample entry:")
    print(df.iloc[0])
