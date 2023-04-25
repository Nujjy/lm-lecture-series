## Language Modeling Lecture Series

Lecture 1: Introduction to Language Modeling

- What is language modeling?
- The N-Gram Language Model
- Maximum Liklihood Estimation and Bigram Statistics
- Building a Character Level Bigram Language Model.
- Estimating the Loss

Lecture 2: Bigram Language Model Optimization (Continued)

- Computing the loss for a Bigram Language Model using the MLE.
- Deriving the loss using gradient descent.

Lecture 3: nn.Linear vs nn.Embeddings vs nn.Parameters
- Writing a n-gram nn.Module in Pytorch
- Looking at multiple different ways to implement the model in Pytorch using nn.Parameter(), nn.Linear(), nn.Embeddings()
- Implementing the Bigram and Trigram Models
- 
Lecture 4: Neural Language Models
- Implementing the Yoshua and Bengio paper in Pytorch.

Lecture 5: Word-2-Vec
- Implementing word-2-vec from scratch.
- Skip-gram Model
- C-bow Model
- The problem with OOV words.

Lecture 5: Tokenizers
- An overview of Fasttext
- Building your own tokenizer from scratch
- Byte Pair Encoding

Lecture 6: Recurrent Neural Network and Additive Attention
- An introduction to recurrent neural networks.
- An overview of the Bahdanau et al. (2014) paper.

Lecture 7: Transformers (More than meets the eye)
- An overview of self attention and some intuition behind it.
- An implementation of the self-attention block
- An overview of the Vaswani et al Paper
- Implementing the entire attention block as an nn.Module from scratch

Lecture 8: BERT and GPT2
- An overview of the devlin et al paper
- An overview of the GPT2 paper.
- Implementing BERT and GPT2 from scratch.

Lecture 9: A survey of how language models have been used for NLP tasks
- Text Classification
- Question Answering

Lecture 10: Text-To-Text Generation and the T5 family of models

