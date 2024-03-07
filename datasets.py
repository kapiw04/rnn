"""
This file contains the dataset classes for the project.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import tensor
from hyperparameters import SEQUENCE_LEGTH, BATCH_SIZE
from my_utils import char_to_idx

filename = r"texts/tinyshakespeare.txt"

with open(filename, 'r') as file:
    text = file.read()

vocab = sorted(set(text))
vocab_length = len(vocab)

class CharDataset(Dataset):
    def __init__(self, text, seq_length, vocab):
        self.text = text
        self.seq_length = seq_length
        self.vocab = vocab
        self.vocab_length = len(vocab)

    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, index):
        sequence_chars = self.text[index:index+self.seq_length]
        # Inside __getitem__ of CharDataset
        next_char = self.text[index + self.seq_length]  # This should be a character, not an index
        next_char_index = char_to_idx(next_char, self.vocab)  # Convert character to index


        sequence = torch.zeros(self.seq_length, self.vocab_length, dtype=torch.float32)
        for i, char in enumerate(sequence_chars):
            sequence[i, char_to_idx(char, self.vocab)] = 1.0

        next_char_index = char_to_idx(next_char, self.vocab)
        nc_tensor = tensor(next_char_index, dtype=torch.long)

        return sequence, nc_tensor
    
train_dataset = CharDataset(text, SEQUENCE_LEGTH, vocab)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)