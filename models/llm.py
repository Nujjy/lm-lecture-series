from torch import nn

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)


