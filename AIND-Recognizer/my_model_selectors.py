import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        all_n_components = range(self.min_n_components, self.max_n_components+1)
        all_scores = []
        N, d = self.X.shape
        for n_components in all_n_components:
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                bic = -2 * logL + (n_components*(n_components-1) + 2*d*n_components) * np.log(N)
                all_scores.append(bic)
            except ValueError:
                # eliminate non-viable models from consideration
                all_scores.append(float("inf"))
        return self.base_model(all_n_components[np.argmin(all_scores)])


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = set(self.words.keys())
        other_words.discard(self.this_word)
        dic_scores = []

        for n_components in range(self.min_n_components, self.max_n_components):
            # apparently, the hmmlearn library is not able to train or score all models,
            # so we just skip the models for which we get an exception
            try:
                hmm_model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                log_likelihood = hmm_model.score(self.X, self.lengths)  # log likelihood
                log_antilikelihood = np.average([hmm_model.score(*self.hwords[w]) for w in other_words])
                dic = log_likelihood - log_antilikelihood
                dic_scores.append((dic, n_components))
            except ValueError:
                if self.verbose:
                    print("hmmlearn failed to train model on {} with {} states".format(self.this_word, num_states))

        # Use the number of components that maximizes the bic score
        best_num_components = self.min_n_components
        if len(dic_scores) > 0:
            best_num_components = max(dic_scores)[1]

        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        all_n_components = range(self.min_n_components, self.max_n_components+1)
        split_method = KFold()
        all_scores = []
        for n_components in all_n_components:
            try:
                scores = []
                if len(self.sequences) > 2:
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Prepare training sequences
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        # Prepare testing sequences
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        model = self.base_model(n_components)
                        scores.append(model.score(X_test, lengths_test))
                    all_scores.append(np.mean(scores))
                else:
                    model = self.base_model(n_components)
                    all_scores.append(model.score(self.X, self.lengths))
            except ValueError:
                # eliminate non-viable models from consideration
                all_scores.append(float("-inf"))
        return self.base_model(all_n_components[np.argmax(all_scores)])
