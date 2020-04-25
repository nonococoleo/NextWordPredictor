# https://stackabuse.com/python-for-nlp-developing-an-automatic-text-filler-using-n-grams/

from data import *


def build_model(n, file):
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
    print("Train File: " + file)
    print("N: " + str(n))
    return ngrams


def chooseFromDist(pos):
    # pos = {'A': Decimal("0.3"), 'B': Decimal("0.4"), 'C': Decimal("0.3")}
    best_word = "the"
    best_prob = 0
    for k, p in pos.items():
        if best_prob < p:
            best_prob = p
            best_word = k
    return best_word


def run_model(n, ngrams, given_words, num_to_predict):
    curr_sequence = ' '.join(given_words[-n:])
    #print()
    output = curr_sequence
    for i in range(num_to_predict):
        if curr_sequence not in ngrams.keys():
            # todo: find a way to select next word even if the given sequence not exists in model
            # choose most often occurred words or change the given sequence to a similar one
            return "the"
        possible_words = ngrams[curr_sequence]
        next_word = chooseFromDist(possible_words)
        if num_to_predict == 1:
            return next_word
        output += ' ' + next_word
        seq_words = nltk.word_tokenize(output)
        curr_sequence = ' '.join(seq_words[len(seq_words) - n:len(seq_words)])
    return curr_sequence
    #print(output)


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
        word = run_model(n,ngrams,words_tokens[i:i+n],1)
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


n = 1
model = build_model(n, "corpus/train_all.pkl")
# given_previous_words = ["May", "last", "year", ","]
# number_of_next_words_to_predict = 50
# run_model(n, model, given_previous_words, number_of_next_words_to_predict)
test_file = "corpus/test_business.pkl"
test_model(n, model, test_file)

''' 
Test File: corpus/test_business.pkl
Total Words: 19476
Correct Count: 3267
Wrong Count: 16209
Accuracy: 0.16774491682070242
'''

