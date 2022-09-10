import generate
import train
import numpy as np


class NGram:
    """
    Ordinary N-gram language model with flexible prefix length.

    Parameters
    ----------
    prefix_len : int, default=2
        Describes a length of a prefix on which the model will be learning. If not set, uses a prefix of length 2,
        which corresponds to an ordinary Trigram model.

    Notes
    -----
    It is better to add methods save_model() and load_model(), because it is way better to load model only once and
    save model only if needed. In general, it adds way more flexibility, when using the class.

    """

    def __init__(self, prefix_len=2):
        self.model_dict = None
        self.input_dir = None
        self.model = ''
        self.prefix_len = prefix_len
        self.fake_word = '<s>'
        self.join_symbol = '<>'

    def fit(self, input_dir=None):
        """
        Fit N-gram model.

        Parameters
        ----------
        input_dir : string, default=None
            Path to a directory where a training data is located. If not provided input training data from stdin.

        """

        if input_dir is not None and input_dir[-1] != '/':
            input_dir += '/'
        self.input_dir = input_dir
        
        text = train.get_fit_data(self.input_dir)
        data = train.tokenize_data(text, self.prefix_len, self.fake_word)
        fit_dict = train.fit(data, self.prefix_len, self.join_symbol)
        self.model_dict = train.find_most_probable(fit_dict, self.fake_word)
        self.model_dict['prefix_len'] = self.prefix_len

    def save(self, model=None):
        """
        Save N-gram model in .pickle format.

        Parameters
        ----------
        model : string, default=None
            Path to a directory where the model will be saved.

        """

        if model is not None and model[-1] != '/':
            model += '/'
        self.model = model

        save_path = f'{self.model}model.pickle'
        train.save_model(self.model_dict, save_path)

    def load(self, model=None):
        """
        Load N-gram model.

        Parameters
        ----------
        model : string, default=None
            Path to a directory where the model is located.

        Notes
        -----
        If loaded model's prefix_len differs from self.prefix_len, set self.prefix_len to the one in the loaded model.

        """

        if model is None:
            model = f'{self.model}model.pickle'
        self.model_dict = generate.load_model(model)

        if self.model_dict['prefix_len'] != self.prefix_len:
            print("Loaded model prefix length does not match current prefix length")
            print("Setting current prefix length to the one in loaded model")
            self.prefix_len = self.model_dict['prefix_len']

    def generate(self, prefix='', _length=1):
        """
        Generate sequence of words, using given prefix with N-gram model. No more than _length of words will be created.
        If an end of a sentence will be found, the generation will stop (I made it so, because sometimes model falls
        into loop and the sentence gets ruined).

        Parameters
        ----------
        prefix : string, default=''
            Given prefix. If not provided, use random word in loaded model.

        _length : int, default=1
            A number of words that will be generated.

        Returns
        -------
        predicted_words : {array}

        Notes
        -----
        Given prefix length may differ from self.prefix_len. In that case only last self.prefix_len words will be used,
        or the fake ones will be added.

        """

        if not prefix:
            while 1:
                index = np.random.randint(0, len(self.model_dict['fit_dict'].keys()))
                key = list(self.model_dict['fit_dict'].keys())[index]
                prefix = self.model_dict['fit_dict'][key]
                if prefix != self.fake_word:
                    break
        words = train.normalize_sentence(prefix)
        predicted_words = []
        for i in range(_length):
            norm_prefix = generate.normalize_prefix(words, self.prefix_len, self.fake_word)
            predicted_word = generate.predict(norm_prefix, self.model_dict, self.join_symbol)
            if predicted_word == self.fake_word:
                predicted_words.append('.')
                break
            words.append(predicted_word)
            predicted_words.append(predicted_word)
        return predicted_words
