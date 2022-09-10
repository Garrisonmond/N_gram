import pickle
import argparse
import numpy as np

#PLEASE READ README FILE
#PLEASE READ README FILE
#PLEASE READ README FILE

def get_args():
    """
    Get arguments from user using argparse.

    Returns
    -------
    prefix : string
    model : string
    length : int

    """

    parser = argparse.ArgumentParser(description='Get arguments to predict using N-gram model.')
    parser.add_argument('--prefix', type=str, help='Prefix')
    parser.add_argument('--model', type=str, help='Path where the trained model is located', required=True)
    parser.add_argument('--length', type=int, help='The length of sequence to generate', required=True)
    args = parser.parse_args()
    print(args)
    prefix, model, length = args.prefix, args.model, args.length

    return prefix, model, length


def load_model(model):
    """
    Load model.

    Parameters
    ----------

    model : string
        Save path, where the model will be saved.

    Returns
    -------
    model : {dict}

    """

    with open(model, mode='rb') as handle:
        model = pickle.load(handle)

    return model


def choose_random_word(model_dict, fake_word):
    """
    Find random word in trained model dictionary.

    Parameters
    ----------

    model_dict : {dict}
        Trained model

    fake_word : string
        Format of fake word that will be added.

    Returns
    -------
    prefix : string

    """

    while 1:
        index = np.random.randint(0, len(model_dict['fit_dict'].keys()))
        key = list(model_dict['fit_dict'].keys())[index]
        prefix = model_dict['fit_dict'][key]
        if prefix != fake_word:
            break
    return prefix


def multiple_split(text, sep=None):  # Я навсякий случай продублировал функции из train.py
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


def normalize_sentence(sentence):  # Я навсякий случай продублировал функции из train.py
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


def normalize_prefix(words, prefix_len, fake_word):
    """
    Normalize given prefix.

    Parameters
    ----------

    words : {array}
        Array of words.

    prefix_len : int
        Length of prefix.

    fake_word : string
        Format of fake word that will be added.

    Returns
    -------
    norm_prefix : {dict}

    """

    if len(words) > prefix_len:
        return words[-prefix_len:]
    else:
        return [fake_word]*(prefix_len - len(words)) + words


def predict(norm_prefix, model, join_symbol):
    """
    Predicts a word with given prefix.

    Parameters
    ----------

    norm_prefix : {array}
        Normalized prefix.

    model : {dict}
        Fitted N-gram model.

    join_symbol : string
        Format of symbol to join prefix.

    Returns
    -------
    predicted_word : string

    Notes
    -----

    If given prefix is not found in the model dictionary, return most probable word overall.

    """

    joined_prefix = join_symbol.join(norm_prefix)
    if joined_prefix in model['fit_dict'].keys():
        return model['fit_dict'][joined_prefix]
    else:
        return model['most_popular']


def main():
    fake_word = '<s>'
    join_symbol = '<>'

    prefix, model, length = get_args()
    print(prefix, model, length)
    model_dict = load_model(model)
    prefix_len = model_dict['prefix_len']
    if not prefix:
        prefix = choose_random_word(model_dict, fake_word)
    words = normalize_sentence(prefix)
    predicted_words = []
    for i in range(length):
        norm_prefix = normalize_prefix(words, prefix_len, fake_word)
        predicted_word = predict(norm_prefix, model_dict, join_symbol)
        if predicted_word == fake_word:  # Во избежания зацикливания префиксов, остановим модель на найденной точке
            predicted_words.append('.')
            break
        words.append(predicted_word)
        predicted_words.append(predicted_word)

    print(predicted_words)


if __name__ == '__main__':
    main()
