# hw4写的HMM我放上来了（之后或许会用到 用不到就删了吧 _(:⁍」∠)_

import numpy as np


def training(train_filename):
    """Train the HMM model with the training file

    Parameters
    ----------
    train_filename : str
        The file location of the training file

    Returns
    -------
    transition
        a np matrix contains transition probabilities from states to states
    emission
        a np matrix contains emission probabilities from states to words
    word_dict
        a dictionary of words as keys and their index used in matrixes as values
    pos_dict
        a dictionary of state as keys and their index used in matrixes as values
    states
        a list of all states
    start
        a np array contains transition probabilities from start state to other states
    """
    # read file
    train_file = open(train_filename, 'r')
    data = train_file.readlines()

    # create pos and word dictionary
    pos_dict = {}
    word_dict = {}
    pos_count = 0
    word_count = 0
    for i in range(len(data)):
        data[i] = data[i].rstrip('\n')
        if data[i] == "":
            continue
        fields = data[i].split('\t')
        word = fields[0]
        pos = fields[1]
        # fill pos dictionary
        if pos not in pos_dict:
            pos_dict[pos] = pos_count
            pos_count += 1
        # fill word dictionary
        if word not in word_dict:
            word_dict[word] = word_count
            word_count += 1

    # create transition matrix
    transition = np.zeros((pos_count, pos_count))
    # create emission matrix
    emission = np.zeros((pos_count, word_count))
    # create start transition list
    start = np.zeros(pos_count)
    prev_pos = ""
    for i in range(len(data)):
        data[i] = data[i].rstrip('\n')
        if data[i] == "":
            prev_pos = ""
            continue
        fields = data[i].split('\t')
        # fill transition matrix
        cur_pos = fields[1]
        cur_pos_idx = pos_dict[cur_pos]
        if prev_pos != "":
            prev_pos_idx = pos_dict[prev_pos]
            transition[prev_pos_idx][cur_pos_idx] += 1
        else:
            # fill start transition list
            start[cur_pos_idx] += 1
        prev_pos = cur_pos
        # fill emission matrix
        cur_word = fields[0]
        word_idx = word_dict[cur_word]
        pos_idx = pos_dict[cur_pos]
        emission[pos_idx][word_idx] += 1

    # compute transition probabilities
    for row in range(transition.shape[0]):
        if np.sum(transition[row]) != 0:
            transition[row] /= np.sum(transition[row])
    # compute emission probabilities
    for row in range(emission.shape[0]):
        if np.sum(emission[row]) != 0:
            emission[row] /= np.sum(emission[row])
    # compute start transition probabilities
    if np.sum(start) != 0:
        start /= np.sum(start)

    # create list of states
    states = [None] * len(pos_dict)
    for k,v in pos_dict.items():
        states[v] = k

    return transition, emission, word_dict, pos_dict, states, start


def viterbi_algo(obs, transition, emission, word_dict, pos_dict, states, start):
    """Run Viterbi algorithm (Reference: Viterbi algorithm From Wikipedia)

    Parameters
    ----------
    obs : list[str]
        The sentence need to be tagged
    transition: np matrix
        Transition probabilities from states to states
    emission: np matrix
        Emission probabilities from states to words
    word_dict: dictionary
        A dictionary of words as keys and their index used in matrixes as values
    pos_dict: dictionary
        A dictionary of state as keys and their index used in matrixes as values
    states: list
        A list of all states
    start: np array
        A np array contains transition probabilities from start state to other states

    Returns
    -------
    path
        Tags for the sentence
    """
    viterbi = [{}]
    # initialize start probabilities
    for cur_state in states:
        start_idx = pos_dict[cur_state]
        if obs[0] in word_dict:
            row = pos_dict[cur_state]
            col = word_dict[obs[0]]
            viterbi[0][cur_state] = {"prob": start[start_idx] * emission[row][col], "prev": None}
        # Viterbi decoding with a uniform probability for unknown words
        else:
            emit_prob = 1/len(word_dict)
            viterbi[0][cur_state] = {"prob": start[start_idx] * emit_prob, "prev": None}

    for i in range(1, len(obs)):
        viterbi.append({})
        for cur_state in states:
            # initialize max_tr_prob and prev_st_selected
            row = pos_dict[states[0]]
            col = pos_dict[cur_state]
            max_prob = viterbi[i - 1][states[0]]["prob"] * transition[row][col]
            max_prev_state = states[0]
            # find out previous state with max probability
            for prev_state in states[1:]:
                row = pos_dict[prev_state]
                col = pos_dict[cur_state]
                cur_prob = viterbi[i - 1][prev_state]["prob"] * transition[row][col]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_prev_state = prev_state
            # compute total probability (by multiplying emit probability)
            if obs[i] in word_dict:
                row = pos_dict[cur_state]
                col = word_dict[obs[i]]
                max_prob = max_prob * emission[row][col]
            # Viterbi decoding with a uniform probability for unknown words
            else:
                emit_prob = 1 / len(word_dict)
                max_prob = max_prob * emit_prob
            # record data
            viterbi[i][cur_state] = {"prob": max_prob, "prev": max_prev_state}

    path = []
    max_prob = 0.0
    best_st = None

    # Get the best state and its backtrack
    for cur_state, data in viterbi[-1].items():
        best_st = ""
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = cur_state
        if best_st == "":
            # assign '.' for the last word of each sentence
            best_st = "."
    path.append(best_st)
    previous = best_st

    # Follow the backtrack till the first observation
    for i in range(len(viterbi) - 2, -1, -1):
        path.insert(0, viterbi[i + 1][previous]["prev"])
        previous = viterbi[i + 1][previous]["prev"]

    return path


def testing(test_filename, transition, emission, word_dict, pos_dict, states, start):
    """Run the trained HHM model of the test file

    Parameters
    ----------
    train_filename : str
        The file location of the training file
    transition: np matrix
        a np matrix contains transition probabilities from states to states
    emission: np matrix
        a np matrix contains emission probabilities from states to words
    word_dict: dictionary
        a dictionary of words as keys and their index used in matrixes as values
    pos_dict: dictionary
        a dictionary of state as keys and their index used in matrixes as values
    states: list
        a list of all states
    start: np array
        a np array contains transition probabilities from start state to other states
    """
    # read file
    test_file = open(test_filename, 'r')
    test = test_file.readlines()

    # put each sentence in a list
    obs = []
    sentences = []
    for i in range(len(test)):
        test[i] = test[i].rstrip('\n')
        if test[i] == "":
            if obs:
                sentences.append(obs)
                obs = []
                continue
        word = test[i]
        obs.append(word)
    if obs:
        sentences.append(obs)

    # tag each sentence
    tags = []
    for i in range(len(sentences)):
        cur_sentence = viterbi_algo(sentences[i], transition, emission, word_dict, pos_dict, states, start)
        tags = tags + cur_sentence

    # add the tags to the original file
    for i in range(len(test)):
        test[i] = test[i].rstrip('\n')
        if test[i] == "":
            test[i] = '\n'
            continue
        pos = tags.pop(0)
        test[i] = test[i] + '\t' + pos + '\n'
    with open("wsj_23.pos", 'w') as test_file:
        test_file.writelines(test)


if __name__ == '__main__':
    transition, emission, word_dict, pos_dict, states, start = training("WSJ_POS_CORPUS_FOR_STUDENTS/WSJ_02-21.pos")
    testing("WSJ_POS_CORPUS_FOR_STUDENTS/wsj_23.words", transition, emission, word_dict, pos_dict, states, start)
