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


class UnigramLM(nn.Module):

    def __init__(self, vocab_size):
        super(UnigramLM, self).__init__()
        self.logits = nn.Parameter(torch.rand(vocab_size))

    def forward(self, inputs):
        return self.logits.expand(inputs.size(0), -1)

class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=10, ngram=5):
        super(NgramLanguageModel, self).__init__()
        self.context_len = ngram - 1
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(self.context_len * embedding_dim, vocab_size)

    def forward(self, inputs):
        Z = self.embeddings(inputs[:, :self.context_len])
        z1 = Z.view((Z.size(0)), -1)
        o = self.linear1(z1)
        return o


class EverygramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, ngram=5):
        super(EverygramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.ngram_models = nn.ModuleDict({
            f"1-gram": UnigramLM(vocab_size),
            **{f"{i}-gram": NgramLanguageModel(vocab_size, embedding_dim, i) for i in range(2, ngram+1)}
        })
        self.log_lambdas = nn.Parameter(torch.randn(len(self.ngram_models)))

    def forward(self, inputs):
        # Compute lambdas
        lambdas = nn.functional.softmax(self.log_lambdas, dim=0)

        # Compute logits from each model and sum them
        logits = torch.zeros((inputs.size(0), self.vocab_size))
        for lambda_, (_, model) in zip(lambdas, self.ngram_models.items()):
            logits += (lambda_ * model(inputs))  # model(inputs) should return logits

        return logits

