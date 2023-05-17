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
        return self.logits

class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=10, ngram=5):
        super(NgramLanguageModel, self).__init__()
        self.ngram = ngram
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear1 = nn.Linear(ngram * embedding_dim, vocab_size)

    def forward(self, inputs):
        Z = self.embeddings(inputs[:,:self.ngram])
        z1 = Z.view((Z.size(0)),-1)
        o = self.linear1(z1)
        return o


class EverygramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, ngram=5):
        super(EverygramLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.ngram_models = nn.ModuleDict({
            "1-gram": UnigramLM(vocab_size),
            **{f"{i+2}-gram": NgramLanguageModel(vocab_size, embedding_dim, i+2) for i in range(ngram-1)}
        })
        self.log_lambdas = nn.Parameter(torch.randn(len(self.ngram_models)))

    def forward(self, inputs):
        # Compute log lambdas
        log_lambdas = nn.functional.log_softmax(self.log_lambdas, dim=0)

        # Compute log probabilities from each model and sum them
        log_prob = torch.zeros((inputs.size(0), self.vocab_size))
        for log_lambda, (_, model) in zip(log_lambdas, self.ngram_models.items()):
            log_prob += (log_lambda + model(inputs))

        return log_prob

