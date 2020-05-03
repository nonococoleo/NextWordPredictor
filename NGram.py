# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
# https://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf
# https://rpubs.com/leomak/TextPrediction_KBO_Katz_Good-Turing

# -*- coding: utf-8 -*-
import argparse
from data import *

# Default Values
n = 3
training_file = "corpus/train_all.pkl"
test_file = "corpus/test_all.pkl"
output_file = "output.txt"
backoff = True
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, choices=[1, 2, 3, 4, 5], help="N for NGram")
parser.add_argument("--training_file", help="Training file path, default: corpus/train_all.pkl")
parser.add_argument("--test_file", help="Test file path, default: corpus/test_all.pkl")
parser.add_argument("--output_file", help="Output filename, default: output.txt")
parser.add_argument("--backoff", help="Using backoff model, default: True")

args = parser.parse_args()
if args.__dict__["N"] is not None:
    n = args.__dict__["N"]
if args.__dict__["training_file"] is not None:
    training_file = args.__dict__["training_file"]
if args.__dict__["test_file"] is not None:
    test_file = args.__dict__["test_file"]
if args.__dict__["output_file"] is not None:
    output_file = args.__dict__["output_file"]
if args.__dict__["backoff"] is not None:
    if args.__dict__["backoff"] == "False":
        backoff = False

MOST_COMMON_WORD = "the"


def build_model(n, file):
    ngrams_backoff = []
    data = load_data(file)
    text = ""
    for lines in data:
        text += lines + "\n"
    corpus = clean(text)
    data, words_tokens = pad(corpus)
    vocab_size = len(set(words_tokens))
    for cur_n in range(1, n+1):
        ngrams = {}
        for i in range(len(words_tokens) - cur_n):
            seq = ' '.join(words_tokens[i:i + cur_n])
            if seq not in ngrams.keys():
                ngrams[seq] = {}
            if words_tokens[i + cur_n] not in ngrams[seq].keys():
                ngrams[seq][words_tokens[i + cur_n]] = 1
            else:
                ngrams[seq][words_tokens[i + cur_n]] += 1
        for seq in ngrams.keys():
            count = 0
            for words in ngrams[seq].keys():
                count += ngrams[seq][words]
            for words in ngrams[seq].keys():
                ngrams[seq][words] /= count
        ngrams_backoff.append(ngrams)
    return ngrams_backoff


def chooseFromDist(pos):
    # pos = {'A': Decimal("0.3"), 'B': Decimal("0.4"), 'C': Decimal("0.3")}
    best_word = MOST_COMMON_WORD
    best_prob = 0
    for k, p in pos.items():
        if best_prob < p:
            best_prob = p
            best_word = k
    return best_word


def run_model(n, ngram_backoff, given_words, backoff):
    sequence = ' '.join(given_words[-n:])
    if backoff:
        for ngram in reversed(ngram_backoff):
            if sequence in ngram.keys():
                possible_words = ngram[sequence]
                return chooseFromDist(possible_words)
            else:
                sequence = ' '.join(given_words[-(n - 1):])
    else:
        if sequence in ngram_backoff[-1].keys():
            possible_words = ngram_backoff[-1][sequence]
            return chooseFromDist(possible_words)
    return MOST_COMMON_WORD


def test_model(n, ngrams, test_file, backoff):
    test_data = load_data(test_file)
    test_text = ""
    correctCount = 0
    wrongCount = 0
    #empty = 0
    for lines in test_data:
        test_text += lines + "\n"
    test_corpus = clean(test_text)
    test_data, words_tokens = pad(test_corpus)

    for i in range(len(words_tokens)-n):
        word = run_model(n,ngrams,words_tokens[i:i+n], backoff)
        if words_tokens[i+n] == word:
            correctCount += 1
        else:
            wrongCount += 1

    file = open(output_file, 'a+')
    file.write("N: " + str(n) + "\n")
    file.write("Train File: " + training_file + "\n")
    file.write("Test File: " + test_file+"\n")
    file.write("Backoff: " + str(backoff)+"\n")
    file.write("Total Words: " + str(len(words_tokens) - n)+"\n")
    file.write("Correct Count: " + str(correctCount)+"\n")
    file.write("Wrong Count: " + str(wrongCount)+"\n")
    file.write("Accuracy: " + str(correctCount / (len(words_tokens) - n))+"\n")
    file.write("\n")
    print("N: " + str(n))
    print("Train File: " + training_file)
    print("Test File: " + test_file)
    print("Backoff: " + str(backoff))
    print("Total Words: " + str(len(words_tokens)-n))
    print("Correct Count: " + str(correctCount))
    print("Wrong Count: " + str(wrongCount))
    print("Accuracy: " + str(correctCount / (len(words_tokens) - n)))
    print()


model = build_model(n, training_file)
test_model(n, model, test_file, backoff)

''' 
Train File: corpus/train_all.pkl
N: 1
Test File: corpus/test_business.pkl
Total Words: 19475
Correct Count: 3739
Wrong Count: 15736
Accuracy: 0.19198973042362003
'''

