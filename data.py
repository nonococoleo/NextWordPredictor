import nltk
import os
from pickle import dump,load


def pad(corpus, length=3):
    data = []
    label = []
    for sent in corpus:
        for k in range(0, len(sent)):
            data.append((['<PAD>'] * (length - k)) + sent[max(0, k - length):k])
            label.append(sent[k])
    return data, label


def clean(text):
    res = []
    for i in text.split('\n'):
        for j in nltk.sent_tokenize(i):
            temp = []
            sent = nltk.word_tokenize(j)
            if len(sent) > 2:
                for k in sent:
                    if k.isalpha() or k == ',':
                        temp.append(k)
                temp.append('.')
                res.append(temp)
    return res


def load_dataset(path, category):
    files = os.listdir(path)
    res = []
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        res.append(lines)
    with open(path + category + ".pkl", "rb") as f:
        dump(res, f, -1)

def load_data(path):
    with open(path,'rb') as f:
        data=load(f)
    return data

if __name__ == '__main__':
    data=load_data("corpus/train_business.pkl")
    text = ""
    for lines in data:
        text += lines + "\n"
    corpus = clean(text)
    data,label=pad(corpus)
    print(data[:10])
    print(label[:10])
