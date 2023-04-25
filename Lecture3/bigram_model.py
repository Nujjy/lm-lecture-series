import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

with open('../names.txt', 'r') as f:
    names = f.read().split()

class CharLevelTrigramDataset(Dataset):

    def __init__(self, text, stop_char='.'):
        self.text = text
        self.vocab = [stop_char] + sorted(list(set(''.join(names))))
        self.ctoi = {c: i  for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.X, self.Y = self.precompute_tensors()

    def precompute_tensors(self):
        xs = []
        ys = []
        for n in names:
            n = ['.'] + list(n) + ['.']
            for c1, c2, c3 in zip(n, n[1:], n[2:]):
                ix1, ix2, ix3 = self.ctoi[c1], self.ctoi[c2], self.ctoi[c3]
                xs.append((ix1,ix2))
                ys.append(ix2)
        return torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = self.X[idx]
        target_tensor = self.Y[idx]
        return input_tensor, target_tensor

def collate_fn(batch):
    inputs, targets = batch[0][0], batch[0][1]
    return inputs, targets

dataset = CharLevelTrigramDataset(names)

sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.SequentialSampler(dataset),
    batch_size=1,
    drop_last=False)

dataloader = DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)


class TrigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(TrigramLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, inputs):
        Z = self.embedding(inputs)
        z = Z.view(Z.size(0),-1)

        return z



lr = 10
num_epochs = 1000
model = TrigramLanguageModel(len(dataset.vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    total_loss = 0
    count = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        probs = model(x)
        loss = criterion(probs,y)
        loss.backward()
        optimizer.step()

        total_loss += loss
        count += 1

    print(total_loss / count)