"""Scripts to run the experiments (and view the results)."""

from collections import Counter
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from subprocess import run


def fastText(train, test):
    """Call shell script with fixed fasttext config to train and test.

    Parameters
    ----------
    train : str
        Data id of the set used for training the model.

    test : str
        Data id of the set used to test the model on.

    """
    run(['sh', 'sec4_exp.sh', train, test])


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Standard majority baseline implementation using sklearn classes."""

    def __init__(self):
        """Set label counter."""
        self.y_counter = Counter()

    def fit(self, X, y):
        """Count the labels."""
        for yi in y:
            self.y_counter[yi] += 1

    def predict(self, X):
        """Predict the majority label for the provided data."""
        y_pred = [self.y_counter.most_common(1)[0][0]
                  for _ in range(X.shape[0])]
        return y_pred

    def mb_score(self, data):
        """Score performance for majority baseline prediction."""
        y_true = []
        for d in data:
            dat = d.split(' ')
            y = 'f' if int(dat.pop(0).replace('__label__', '')) == 1 else 'm'
            self.y_counter[y] += 1
            y_true.append(y)
        y_pred = [self.y_counter.most_common()[0][0]] * len(y_true)

        return accuracy_score(y_true, y_pred)


class LexiconGender(object):
    """Weighted lexicon approach by Sap et al.

    For implementation details, see paper below.

    The vocab and weights can be downloaded from
    http://www.wwbp.org/lexica.html. And are assumed to be under
    ./corpora/emnlp14gender.csv.

    Notes
    -----
    This classifier is originally from:

    @article{sap2014developing,
      title={Developing age and gender predictive lexica over social media},
      author={Sap, Maarten and Park, Gregory and Eichstaedt, Johannes C and
              Kern, Margaret L and Stillwell, David and Kosinski, Michal and
              Ungar, Lyle H and Schwartz, H Andrew},
      year={2014},
      publisher={Citeseer}
    }

    """

    def __init__(self):
        """Load lexcion and weights."""
        df = pd.DataFrame.from_csv('./corpora/emnlp14gender.csv')
        lexD = df.to_dict()
        self._intercept = lexD['weight'].pop('_intercept')
        self._weights = dict(lexD['weight'].items())

    def predict(self, tokens):
        """Predict instance according to the vocab weights."""
        val = 0
        for token in tokens:
            val += self._weights.get(token, 0)
        val += self._intercept
        return 'm' if val < 0 else 'f'

    def lex_score(self, data):
        """Score performance of the lexicon classifier given data."""
        y_true = []
        y_pred = []

        for i, d in enumerate(data):
            dat = d.split(' ')
            try:
                y = 'f' if int(dat.pop(0).replace('__label__', '')) == 1 \
                    else 'm'
                y_hat = self.predict([di.lower() for di in dat])
                y_true.append(y)
                y_pred.append(y_hat)
            except ValueError:  # last line
                pass

        return accuracy_score(y_true, y_pred)


if __name__ == '__main__':
    bl = MajorityBaseline()
    lg = LexiconGender()
    for train in ['query', 'plank', 'volkova']:  # , 'twitter' ]
        for test in ['query', 'plank', 'volkova']:  # , 'twitter' ]
            print("\n\n>>> train: {0} \t test: {1}".format(train, test))
            fastText(train + '_gender', test + '_gender')
            fin = open('./data/{0}_gender.test'.format(test)).readlines()
            sscore = round(lg.lex_score(fin), 3)
            bscore = round(bl.mb_score(fin), 3)
            print("\nSap baseline @ test: {}".format(sscore))
            print("Maj baseline @ test: {}".format(bscore))
    print("\n\n")
