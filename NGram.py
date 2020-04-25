#https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/

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
            ngrams[seq] = []
        ngrams[seq].append(words_tokens[i + n])
    return ngrams


def run_model(n=3, ngrams=build_model(), given_words=["Ad","sales","boost"], num_to_predict=50):
    curr_sequence = ' '.join(given_words[-n:])
    print()
    output = curr_sequence
    for i in range(num_to_predict):
        if curr_sequence not in ngrams.keys():
            # todo: find a way to select next word even if the given sequence not exists in model
            break
        possible_words = ngrams[curr_sequence]
        # todo: now is random selection, try to make it probability based
        next_word = possible_words[random.randrange(len(possible_words))]
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words) - n:len(seq_words)])
    print(output)


n = 3
model = build_model(n, "corpus/train_business.pkl")
given_previous_words = ["May", "last", "year", ","]
number_of_next_words_to_predict = 50
run_model(n, model, given_previous_words, number_of_next_words_to_predict)
