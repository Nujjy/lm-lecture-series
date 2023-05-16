import torch
from torch.utils.data import Dataset, DataLoader

class CharLevelBigramDataset(Dataset):

    def __init__(self, text, stop_char='.'):
        self.text = text
        self.vocab = [stop_char] + sorted(list(set(''.join(text))))
        self.ctoi = {c: i  for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.X, self.Y = self.precompute_tensors()

    def precompute_tensors(self):
        xs = []
        ys = []
        for n in self.text:
            n = ['.'] + list(n) + ['.']
            for c1, c2, c3 in zip(n, n[1:], n[2:]):
                ix1, ix2 = self.ctoi[c1], self.ctoi[c2]
                xs.append(ix1)
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = self.X[idx]
        target_tensor = self.Y[idx]
        return input_tensor, target_tensor


class CharLevelTrigramDataset(Dataset):

    def __init__(self, text, stop_char='.'):
        self.text = text
        self.vocab = [stop_char] + sorted(list(set(''.join(text))))
        self.ctoi = {c: i  for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.X, self.Y = self.precompute_tensors()

    def precompute_tensors(self):
        xs = []
        ys = []
        for n in self.text:
            n = ['.'] + list(n) + ['.']
            for c1, c2, c3 in zip(n, n[1:], n[2:]):
                ix1, ix2, ix3 = self.ctoi[c1], self.ctoi[c2], self.ctoi[c3]
                xs.append((ix1,ix2))
                ys.append(ix3)
        return torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = self.X[idx]
        target_tensor = self.Y[idx]
        return input_tensor, target_tensor


class CharLevelNgramDataset(Dataset):
    def __init__(self, text, stop_char='.', context_length=3):
        self.context_length = context_length
        self.text = text
        self.vocab = [stop_char] + sorted(list(set(''.join(names))))
        self.ctoi = {c: i  for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.X, self.Y = self.precompute_tensors()

    def precompute_tensors(self):
        xs = []
        ys = []
        for n in self.text:
            n = ['.'] + list(n) + ['.']
            ######################
            ## INSERT CODE HERE ##
            ######################
        return torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = self.X[idx].squeeze(0)
        target_tensor = self.Y[idx].squeeze(0)
        return input_tensor, target_tensor