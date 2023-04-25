import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

with open('../names.txt', 'r') as f:
    names = f.read().split()

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
        for n in names:
            n = ['.'] + list(n) + ['.']
            for c in zip(n, *[n[i:] for i in range(1,self.context_length)]):
                indices = [self.ctoi[ci] for ci in c]
                xs.append(indices[:-1])
                ys.append(indices[-1])
        return torch.tensor(xs), torch.tensor(ys)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = self.X[idx].squeeze(0)
        target_tensor = self.Y[idx].squeeze(0)
        return input_tensor, target_tensor

def collate_fn(batch):
    inputs, targets = batch[0][0], batch[0][1]
    return inputs, targets

dataset = CharLevelNgramDataset(names, context_length=2)

sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.SequentialSampler(dataset),
    batch_size=len(dataset),
    drop_last=False)

dataloader = DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)


class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim=32, ngram=5):
        super(NgramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear((ngram-1) * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        Z = self.embeddings(inputs)
        z1 = Z.view((Z.size(0)),-1)
        z2 = self.linear1(z1)
        o = self.linear2(z2)
        return o


lr = 0.1
num_epochs = 1000
model = NgramLanguageModel(len(dataset.vocab), embedding_dim=64, ngram=2)
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