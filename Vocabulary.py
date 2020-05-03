import numpy as np
from data import *


class Vocabulary:
    def __init__(self, glove_file, dimension=300):
        """
        initialize vocabulary dict with glove vectors
        :param glove_file: pretrained glove vector file
        :param dimension: dimension of vectors
        """
        self.glove = self.load_glove(glove_file)
        self.dims = dimension
        print("glove loaded")
        self.idx2word = ['<PAD>', '<UNK>']
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.num_of_words = 2

    def load_glove(self, file_path):
        """
        reading glove file
        :param file_path: path to glove file
        :return: glove vectors
        """
        res = dict()
        with open(file_path) as f:
            for line in f:
                try:
                    token, raw_embedding = line.split(maxsplit=1)
                    res[token] = np.array([float(x) for x in raw_embedding.split()])
                except:
                    pass
        return res

    def get_idx(self, words):
        """
        convert words to indexes
        :param words: a list of words
        :return: a list of indexes
        """
        return [self.word2idx.get(token, 1) for token in words]

    def get_word(self, idxs):
        """
        convert indexes to words
        :param idxs: a list of indexes
        :return: a list of words
        """
        return [self.idx2word[idx if 0 <= idx < self.num_of_words else 1] for idx in idxs]

    def build_word_dict(self, corpus):
        """
        build vocabulary from corpus
        :param corpus: corpus string
        :return: None
        """
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
        print("word dict built")

    def featurize(self, data, labels):
        """
        convert text data to numerical data
        :param data: input words
        :param labels: target words
        :return: numerical data
        """
        input_data = []
        for sent in data:
            ids = self.get_idx(sent)
            input_data.append(ids)

        label_data = self.get_idx(labels)
        return np.array(input_data), np.array(label_data)

    def load_data(self, corpus_file, window_length):
        """
        load corpus from precleaned file
        :param corpus_file: path to corpus file
        :param window_length: input window length
        :return: numerical data
        """
        data = load_corpus(corpus_file)
        text = ""
        for lines in data:
            text += lines + "\n"
        corpus = clean(text)
        return self.featurize(*pad(corpus, window_length))


if __name__ == '__main__':
    from pickle import dump, load

    # app = Vocabulary('glove.6B.300d.txt')
    app = load(open("vocab", "rb"))

    data = load_corpus("corpus/train_business.pkl")
    text = ""
    for lines in data:
        text += lines + "\n"
    corpus = clean(text)
    print("corpus loaded")

    app.build_word_dict(corpus)
    with open("small-vocab", "wb") as f:
        dump(app, f, -1)
    print("model saved")
