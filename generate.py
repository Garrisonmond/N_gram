import pickle


def load_model(load_path):
    """
    Load model.

    Parameters
    ----------

    load_path : string
        Save path, where the model will be saved.

    Returns
    -------
    model : {dict}

    """

    with open(load_path, mode='rb') as handle:
        model = pickle.load(handle)
    return model


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
