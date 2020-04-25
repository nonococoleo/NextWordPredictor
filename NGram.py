# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/

import random
from data import *


def build_model(n=3, file="corpus/train_business.pkl"):
    ngrams = {}
    data = load_data(file)
    text = ""
    for lines in data:
        text += lines + "\n"
    corpus = clean(text)
    data, words_tokens = pad(corpus)
    for i in range(len(words_tokens) - n):
        seq = ' '.join(words_tokens[i:i + n])
        if seq not in ngrams.keys():
            ngrams[seq] = {}
        if words_tokens[i + n] not in ngrams[seq].keys():
            ngrams[seq][words_tokens[i + n]] = 1
        else:
            ngrams[seq][words_tokens[i + n]] += 1
    for seq in ngrams.keys():
        count = 0
        for words in ngrams[seq].keys():
            count += ngrams[seq][words]
        for words in ngrams[seq].keys():
            ngrams[seq][words] /= count
    return ngrams


def chooseFromDist(pos):
    # pos = {'A': Decimal("0.3"), 'B': Decimal("0.4"), 'C': Decimal("0.3")}
    choice = random.random()
    Tp = 0
    for k, p in pos.items():
        Tp += p
        if choice <= Tp:
            return k
    return 'Choose error'


def run_model(n=3, ngrams=build_model(), given_words=["Ad", "sales", "boost"], num_to_predict=50):
    curr_sequence = ' '.join(given_words[-n:])
    #print()
    output = curr_sequence
    for i in range(num_to_predict):
        if curr_sequence not in ngrams.keys():
            # todo: find a way to select next word even if the given sequence not exists in model
            # choose most often occurred words or change the given sequence to a similar one
            break
        possible_words = ngrams[curr_sequence]
        # probability based: done
        next_word = chooseFromDist(possible_words)
        return next_word
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words) - n:len(seq_words)])
    #print(output)

def test_model(n=3,ngrams=build_model(),test_file="corpus/test_business.pkl"):
    test_data = load_data(test_file)
    test_text = ""
    correctCount = 0
    for lines in test_data:
        test_text += lines + "\n"
    test_corpus = clean(test_text)
    test_data, words_tokens = pad(test_corpus)
    for i in range(len(words_tokens)-n):
        if words_tokens[i+n] == run_model(n,ngrams,words_tokens[i:i+n],1):
            correctCount += 1
    print(correctCount/len(words_tokens))
n = 3
model = build_model(n, "corpus/train_business.pkl")
given_previous_words = ["May", "last", "year", ","]
number_of_next_words_to_predict = 50
run_model(n, model, given_previous_words, number_of_next_words_to_predict)
test_file="corpus/test_business.pkl"
test_model(n,model,test_file)