import nltk


def pre(corpus, length=3):
    data = []
    label = []
    for sent in corpus:
        for k in range(0, len(sent)):
            data.append((['<PAD>'] * (length - k)) + sent[max(0, k - length):k])
            label.append(sent[k])
    return data, label


def clean(text):
    res = []
    for i in nltk.sent_tokenize(text):
        for j in i.split(","):
            temp = []
            sent = nltk.word_tokenize(j.lower())
            for k in sent:
                if len(k) > 1 and k.isalpha():
                    temp.append(k)
            res.append(temp)
    return res


if __name__ == '__main__':
    with open("bell.txt") as f:
        lines = f.readlines()
    text = ""
    for i in lines:
        text += " " + i.strip()
    corpus = clean(text)
    data, label = pre(corpus)
    print(data)
    print(label)
    # data, label = pre(text)
    # for i in range(len(data)):
    #     print(data[i], label[i])
