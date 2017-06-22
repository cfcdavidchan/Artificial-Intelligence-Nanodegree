import warnings
from asl_data import SinglesData
from operator import itemgetter


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_id in range(test_set.num_items):
        log_likelihoods = {}
        for word, model in models.items():
            try:
                log_likelihoods[word] = model.score(*test_set.get_item_Xlengths(word_id))
            except ValueError:
                log_likelihoods[word] = float("-inf")
                
        probabilities.append(log_likelihoods)
        guesses.append(max(log_likelihoods.items(), key=itemgetter(1))[0])
        
    return probabilities, guesses
