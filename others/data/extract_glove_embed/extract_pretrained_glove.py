import torch
import numpy as np
from visdial.data.vocabulary import Vocabulary

GLOVE_PATH = '/home/quanguet/datasets/glove/glove.840B.300d.txt'
WORD_COUNT = '/home/quanguet/datasets/visdial/visdial_1.0_word_counts_train.json'
SAVE_PATH = 'embedding_Glove_840_300d.pkl'

# get vocab
vocab = Vocabulary(WORD_COUNT)
word2id = vocab.word2index
id2word = vocab.index2word

# init weights
embed_size = 300
vocab_size = len(id2word)
std = 1/np.sqrt(embed_size)
weights = np.random.normal(0, scale=std, size=[vocab_size, embed_size])
weights = weights.astype(np.float32)

# load weights from glove
with open(GLOVE_PATH, encoding='utf-8', mode="r") as textFile:
    for line in textFile:
        line = line.split()
        word = ''.join(line[:-300])

        idx = word2id.get(word, None)
        if idx is not None:
            weights[idx] = np.array(line[-300:], dtype=np.float32)

# load state_dict
torch_weights = torch.from_numpy(weights)
embeddings = torch.nn.Embedding(len(word2id), 300)
embeddings.weight.data = torch_weights

# save
torch.save(embeddings.state_dict(), SAVE_PATH)