import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, inputs):
        z = self.embedding(inputs)
        return z

class BigramLMParam(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLMParam, self).__init__()
        self.W = nn.Parameter(data=torch.randn(vocab_size, vocab_size)) # Ensure we initialize based on N~(0,1)

    def forward(self, inputs):
        X = F.one_hot(inputs).float()
        Z = X @ self.W
        return Z

class BigramLMLinear(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLMLinear, self).__init__()
        self.linear = nn.Linear(in_features=vocab_size, out_features=vocab_size, bias=False)

    def forward(self, inputs):
        X = F.one_hot(inputs).float()
        Z = self.linear(X)
        return Z


class TrigramLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim=10):
        super(TrigramLM, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear = nn.Linear(in_features=2 * embedding_dim, out_features=vocab_size)

    def forward(self, inputs):
        Z = self.embeddings(inputs)
        z1 = Z.view((Z.size(0)), -1)
        o = self.linear(z1)
        return o

class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, ngram=5):
        super(NgramLanguageModel, self).__init__()
        ###########################
        # INSERT YOUR SOLUTION HERE
        ###########################
        pass

    def forward(self, inputs):
        ###########################
        # INSERT YOUR SOLUTION HERE
        ###########################
        pass
