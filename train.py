import os
import pickle
import numpy as np


def get_fit_data(input_dir):
    """
    Get fit data from user.

    Parameters
    ----------
    input_dir : string, default=None
        Path to a directory where a training data is located. If not provided input training data from stdin.


    Returns
    -------
    text : string

    """

    text = ''
    if input_dir is not None:

        for file in os.listdir(input_dir)[1:]:
            f = open(input_dir + file, mode='r', encoding='utf-8')
            text += f.read()
            if text[-1] not in ['!', '.', '?']:
                text += '.'
    else:
        while 1:
            print("Input text using stdin")
            text += input()

            if text[-1] not in ['!', '.', '?']:
                text += '.'

            print('Do you want to input another text yes/no')
            res = int(input())

            if not res:
                break

    return text


def multiple_split(text, sep=None):
    """
    Split sentence by multiple symbols.

    Parameters
    ----------
    text : string
        Text to split.

    sep : array, default=None
        Array of separator symbols.

    Returns
    -------
    words : {array}

    """

    if sep is None:
        sep = list(np.arange(32, 47 + 1)) + list(np.arange(58, 64 + 1)) + list(np.arange(91, 96 + 1)) + list(
            np.arange(123, 126 + 1))
        sep.append(ord('\n'))
        sep.append(ord('\t'))
        sep.append(ord('—'))
    else:
        sep = list(map(ord, sep))

    words = []
    cur = ''

    for i in text:
        if ord(i) in sep:
            if cur:
                words.append(cur)
            cur = ''
        else:
            cur += i

    if cur:
        words.append(cur)

    return words


def normalize_sentence(sentence):
    """
    Normalizes a given sentence.

    Parameters
    ----------
    sentence : string
        Sentence to normalize.

    Returns
    -------
    words : {array}

    """

    words = []
    for word in multiple_split(sentence):
        if word == '':
            continue
        words.append(word.lower())
    return words


def tokenize_data(text, prefix_len, fake_word):
    """
    Tokenize given text.

    Parameters
    ----------
    text : string
        Text to tokenize.

    prefix_len : int
        Length of prefix.

    fake_word : string
        Format of fake word that will be added.

    Returns
    -------
    data : {array}

    """

    sentences = multiple_split(text, ['.', '!', '?'])[:-1]
    data = []
    for sentence in sentences:
        if sentence == '':
            continue
        words = normalize_sentence(sentence)
        fake_words = [fake_word]*prefix_len
        data.append(fake_words + words + fake_words)

    return data


def fit(data, prefix_len, join_symbol):
    """
    Fit data.

    Parameters
    ----------
    data : {array}
        Data that will be fitted.

    prefix_len : int
        Length of prefix.

    join_symbol : string
        Format of symbol to join prefix.

    Returns
    -------
    fit_dict : {dict}

    """

    fit_dict = dict()

    for i in data:
        for j in range(len(i) - prefix_len):
            key = join_symbol.join(i[j:prefix_len + j])
            value = i[prefix_len + j]

            if key not in fit_dict.keys():
                fit_dict[key] = dict()

            if value in fit_dict[key].keys():
                fit_dict[key][value] += 1
            else:
                fit_dict[key][value] = 1

    return fit_dict


def find_most_probable(fit_dict, fake_word):
    """
    Reformat model dictionary. Find most probable word for evey prefix.

    Parameters
    ----------
    fit_dict : {dict}
        Fitted dictionary.

    fake_word : string
        Format of fake word that will be added.

    Returns
    -------
    model : {dict}

    """

    maximum_total = 0
    most_popular = ''
    model = dict()

    for key in fit_dict.keys():
        maximum = 0
        word = ''

        for value in fit_dict[key].keys():
            if fit_dict[key][value] > maximum:
                word = value
                maximum = fit_dict[key][value]

            if fit_dict[key][value] > maximum_total and value != fake_word:
                most_popular = value
                maximum_total = fit_dict[key][value]

        fit_dict[key] = word

    model['most_popular'] = most_popular
    model['fit_dict'] = fit_dict

    return model


def save_model(model, save_path):
    """
    Save model as .pickle.

    Parameters
    ----------
    model : {dict}
        Fitted model.

    save_path : string
        Save path, where the model will be saved.

    """

    with open(save_path, mode='wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
