# -*- coding: utf-8 -*-
import argparse
from data import *

# Default prediction output
MOST_COMMON_WORD = "the"

# Default Parameters for Testing
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
# -------------------------

# NGram starts here


def build_model(n, file):
    ngrams_backoff = []
    # tokenize
    data = load_corpus(file)
    text = ""
    for lines in data:
        text += lines + "\n"
    corpus = clean(text)
    data, words_tokens = pad(corpus)
    # build 1-gram, 2-gram,..., n-gram
    for cur_n in range(1, n+1):
        ngrams = {}
        for i in range(len(words_tokens) - cur_n):
            seq = ' '.join(words_tokens[i:i + cur_n])
            # store the sequences
            if seq not in ngrams.keys():
                ngrams[seq] = {}
            if words_tokens[i + cur_n] not in ngrams[seq].keys():
                ngrams[seq][words_tokens[i + cur_n]] = 1
            else:
                ngrams[seq][words_tokens[i + cur_n]] += 1
        # compute probabilities
        for seq in ngrams.keys():
            count = 0
            for words in ngrams[seq].keys():
                count += ngrams[seq][words]
            for words in ngrams[seq].keys():
                ngrams[seq][words] /= count
        ngrams_backoff.append(ngrams)
    return ngrams_backoff


# select the word that has the highest probability
def choose_from_dist(pos):
    # pos = {'A': Decimal("0.3"), 'B': Decimal("0.4"), 'C': Decimal("0.3")}
    best_word = MOST_COMMON_WORD
    best_prob = 0
    for k, p in pos.items():
        if best_prob < p:
            best_prob = p
            best_word = k
    return best_word


# use the trained model to predict the next word
def run_model(n, ngram_backoff, given_words, backoff):
    sequence = ' '.join(given_words[-n:])
    # if it is a backoff model
    if backoff:
        # backoff until find an answer
        for ngram in reversed(ngram_backoff):
            if sequence in ngram.keys():
                possible_words = ngram[sequence]
                return choose_from_dist(possible_words)
            else:
                sequence = ' '.join(given_words[-(n - 1):])
    # if it is not a backoff model
    else:
        if sequence in ngram_backoff[-1].keys():
            possible_words = ngram_backoff[-1][sequence]
            return choose_from_dist(possible_words)
    return MOST_COMMON_WORD


def test_model(n, ngrams, test_file, backoff):
    test_data = load_corpus(test_file)
    test_text = ""
    correct_count = 0
    wrong_count = 0
    # tokenize
    for lines in test_data:
        test_text += lines + "\n"
    test_corpus = clean(test_text)
    test_data, words_tokens = pad(test_corpus)
    # compare prediction with correct words
    for i in range(len(words_tokens)-n):
        word = run_model(n, ngrams, words_tokens[i:i+n], backoff)
        if words_tokens[i+n] == word:
            correct_count += 1
        else:
            wrong_count += 1
    # write output to file
    file = open(output_file, 'a+')
    file.write("N: " + str(n) + "\n")
    file.write("Train File: " + training_file + "\n")
    file.write("Test File: " + test_file+"\n")
    file.write("Backoff: " + str(backoff)+"\n")
    file.write("Total Words: " + str(len(words_tokens) - n)+"\n")
    file.write("Correct Count: " + str(correct_count)+"\n")
    file.write("Wrong Count: " + str(wrong_count)+"\n")
    file.write("Accuracy: " + str(correct_count / (len(words_tokens) - n))+"\n")
    file.write("\n")
    print("N: " + str(n))
    print("Train File: " + training_file)
    print("Test File: " + test_file)
    print("Backoff: " + str(backoff))
    print("Total Words: " + str(len(words_tokens)-n))
    print("Correct Count: " + str(correct_count))
    print("Wrong Count: " + str(wrong_count))
    print("Accuracy: " + str(correct_count / (len(words_tokens) - n)))
    print()


# train
model = build_model(n, training_file)
# test
test_model(n, model, test_file, backoff)

