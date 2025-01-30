from torch.utils.data import Dataset, DataLoader
from transformers import MBart50TokenizerFast
import torch

class TranslationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }
    
def tokenize_data(src_texts, tgt_texts, tokenizer, max_length=200):
    model_inputs = tokenizer(
        src_texts, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )

    labels = tokenizer(
        tgt_texts, 
        max_length=max_length, 
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def prepare_datasets(train_src, train_tgt, val_src, val_tgt, test_src, test_tgt, tokenizer, device, batch_size=16):
    train_encodings = tokenize_data(train_src, train_tgt, tokenizer)
    val_encodings = tokenize_data(val_src, val_tgt, tokenizer)
    test_encodings = tokenize_data(test_src, test_tgt, tokenizer)

    # Move tokenized inputs to the selected device
    train_encodings = {key: torch.tensor(val).to(device) for key, val in train_encodings.items()}
    val_encodings = {key: torch.tensor(val).to(device) for key, val in val_encodings.items()}
    test_encodings = {key: torch.tensor(val).to(device) for key, val in test_encodings.items()}

    train_dataset = TranslationDataset(train_encodings)
    val_dataset = TranslationDataset(val_encodings)
    test_dataset = TranslationDataset(test_encodings)

    return train_dataset, val_dataset, test_dataset