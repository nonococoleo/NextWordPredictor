# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/
# https://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf

from data import *

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
    print("Train File: " + file)
    print("N: " + str(n))
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


def run_model(n, ngram_backoff, given_words):
    sequence = ' '.join(given_words[-n:])
    for ngram in reversed(ngram_backoff):
        if sequence in ngram.keys():
            possible_words = ngram[sequence]
            return chooseFromDist(possible_words)
        else:
            sequence = ' '.join(given_words[-(n-1):])
    return MOST_COMMON_WORD


def test_model(n, ngrams, test_file):
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
        word = run_model(n,ngrams,words_tokens[i:i+n])
        if words_tokens[i+n] == word:
            correctCount += 1
        # elif word == "":
        #     empty += 1
        else:
            wrongCount += 1

    print("Test File: " + test_file)
    print("Total Words: " + str(len(words_tokens)-n))
    print("Correct Count: " + str(correctCount))
    # print(empty)
    print("Wrong Count: " + str(wrongCount))
    print("Accuracy: " + str(correctCount / (len(words_tokens) - n)))


n = 2
model = build_model(n, "corpus/train_all.pkl")
# given_previous_words = ["May", "last", "year", ","]
# run_model(n, model, given_previous_words)
test_file = "corpus/test_business.pkl"
test_model(n, model, test_file)

''' 
Test File: corpus/test_business.pkl
Total Words: 19476
Correct Count: 3267
Wrong Count: 16209
Accuracy: 0.16774491682070242
'''

