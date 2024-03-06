"""
This file contains the dataset classes for the project.
"""
from torch.utils.data import Dataset
from torch import tensor
from hyperparameters import SEQUENCE_LEGTH

filename = r"texts/tinyshakespeare.txt"

with open(filename, 'r') as file:
    text = file.read()
    
class CharTextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.vocab = sorted(set(text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, index):
        sequence = [self.char_to_idx[char] for char in self.text[index:index+self.seq_length]]
        next_char = self.char_to_idx[self.text[index+self.seq_length]]

        seq_tensor = tensor(sequence)
        nc_tensor = tensor(next_char)

        return seq_tensor, nc_tensor
    
shakespeare_dataset = CharTextDataset(text, SEQUENCE_LEGTH)