import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from Lecture3.datasets import CharLevelBigramDataset, CharLevelTrigramDataset
from Lecture3.ngram_model import BigramLM, BigramLMParam, BigramLMLinear, TrigramLM

def collate_fn(batch):
    inputs, targets = batch[0][0], batch[0][1]
    return inputs, targets

with open('../names.txt', 'r') as f:
    names = f.read().split()


dataset = CharLevelTrigramDataset(names)

sampler = torch.utils.data.sampler.BatchSampler(
    torch.utils.data.sampler.SequentialSampler(dataset),
    batch_size=len(dataset),
    drop_last=False)

dataloader = DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)


lr = 1
num_epochs = 1000
model = TrigramLM(len(dataset.vocab))
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