import numpy as np
from data import *


class Vocabulary:

    def __init__(self, embeddings_file, dimension=300):
        self.glove = self.load_glove(embeddings_file)
        self.dims = dimension
        print("glove loaded")
        self.idx2word = ['<PAD>', '<UNK>']
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.num_of_words = 2

    def load_glove(self, glove_path):
        res = dict()
        with open(glove_path) as f:
            for line in f:
                try:
                    token, raw_embedding = line.split(maxsplit=1)
                    embedding = np.array([float(x) for x in raw_embedding.split()])
                    res[token] = embedding
                except:
                    pass
        return res

    def get_idx(self, words):
        return [self.word2idx.get(token, 1) for token in words]

    def get_word(self, idxs):
        return [self.idx2word[idx if 0 <= idx < self.num_of_words else 1] for idx in idxs]

    def load_corpus(self, corpus):
        self.idx2word = ['<PAD>', '<UNK>']
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.num_of_words = 2
        res = []
        res.append(np.zeros(self.dims))
        res.append(np.random.rand(self.dims))
        for i in corpus:
            for word in i:
                if word in self.glove:
                    if word not in self.idx2word:
                        self.idx2word.append(word)
                        self.word2idx[word] = self.num_of_words
                        self.num_of_words += 1
                    res.append(self.glove[word])
        self.embeddings = np.array(res)
        print("corpus loaded")

    def featurize(self, data, labels):
        text_data = []
        for sent in data:
            ids = self.get_idx(sent)
            text_data.append(ids)

        label_data = self.get_idx(labels)
        return np.array(text_data), np.array(label_data)

    def load_data(self, file, window_length):
        with open(file) as f:
            lines = f.readlines()
        text = ""
        for i in lines:
            text += " " + i.strip()
        corpus = clean(text)
        return self.featurize(*pre(corpus, window_length))


if __name__ == '__main__':
    from pickle import dump, load

    # app = Vocabulary('glove.6B.300d.txt')
    app = load(open("vocab", "rb"))

    with open("train.txt") as f:
        lines = f.readlines()
    text = ""
    for i in lines:
        text += " " + i.strip()
    corpus = clean(text)
    print("corpus built")

    app.load_corpus(corpus)

    with open("vocab", "wb") as f:
        dump(app, f, -1)
    print("model saved")
