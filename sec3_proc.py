"""Scripts to run the preprocessing and information parts of the paper."""

import json
# from langdetect import detect  # --- see line 165
from sec3_data import DB
from sec3_data import reconstruct_ids
from sec3_data import log
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
import spacy


class AnnotationStats(object):
    """Factory to post annotation metrics from the original paper.

    Parameters
    ----------
    paper : bool
        If true, limits the amount of annotations to those that were available
        during the writing of the paper (1,456 annotations). If false, uses all
        of the information.

    labels : bool
        If true, output general stats regarding the annotations (amount of
        instances, males, females, other, unsure, and missing).

    bots : bool
        If true, output the percentage of bots in the sample.

    agreement : bool
        Calculate the agreement accuracy between the manually provided labels,
        and those given by the distant query method.

    interrater : bool
        If true, include the Fleisch' Kappa agreement score between the first
        and the second annotater if paper is true, otherwise for all three.

    Parameters
    ----------
    annotations : object
        Loaded JSON file for the annotated twitter profiles from our paper.

    paper : bool
        See parameter description.

    stats : dict
        Dictionary representation for the information required to output the
        statistics.

    args : dict
        Class arguments (kwargs).

    """

    def __init__(self, **kwargs):
        """Open query database with annotations."""
        self.annotations = json.load(open('./corpora/query-gender.json')
                                     )['annotations']
        self.paper = kwargs.get('paper', True)
        if self.paper:
            print("PLEASE NOTE: this information is for the annotation part " +
                  "only.\nThe models were trained on _all_ data.\n\n"
                  "Furthermore, v1 of the paper included kappa scores for " +
                  "'not sure'\nannotations, which should have been excluded " +
                  "(kappa = 0.90 then).\n")
            self.annotations = {k: v for k, v in self.annotations.items()
                                if int(k) < 210040000}
        else:
            print("PLEASE NOTE: any statistics dealing with annotations in " +
                  "the paper\ndeal with a subset, reproducable by setting " +
                  "paper=True.\n\n")
        self.stats = dict({'bots': 0, 'total': 0, 'm': 0, 'f': 0, 'o': 0,
                           '-': 0, '0': 0, 'distant': [], 'hand': [],
                           'ann': []})
        self.calculate_stats()
        self.args = kwargs

    def calculate_stats(self):
        """Caclulate all relevant stats."""
        raters = {'ann1': [], 'ann2': [], 'ann3': []}
        for line in self.annotations.values():
            if line['bot'] == 'True':
                self.stats['bots'] += 1
            if line['majority'] in self.stats:
                self.stats[line['majority']] += 1
            if line['majority'] in ['m', 'f'] and line['query_label2'] != '0':
                self.stats['distant'].append(line['query_label2'])
                self.stats['hand'].append(line['majority'])
            for i in range(1, 4):
                raters['ann' + str(i)].append(line['ann' + str(i)])
            self.stats['total'] += 1
        self.stats['raters'] = raters

    def _kappa(self, raters, y1, y2):
        ignore = ('0', 'o') if self.paper else ('0', 'o', '-')
        simsc = [(x, y) for x, y in zip(raters[y1],
                                        raters[y2])
                 if x not in ignore and y not in ignore]
        return len(simsc), cohen_kappa_score(*zip(*simsc))

    def report(self):
        """Report stats according to args provided in class init."""
        s = self.stats
        a = self.args
        print("Amount of instances \t", len(self.annotations))
        if a.get('labels'):
            print("Amount of males \t", s['m'])
            print("Amount of females \t", s['f'])
            print("Amount of other \t", s['o'])
            print("Amount of unsure \t", s['-'])
            print("Amount of missing \t", s['0'])
        if a.get('bots'):
            print("Percentage of bots \t",
                  round(s['bots'] / s['total'] * 100, 1), '%')
        if a.get('agreement'):
            acc = accuracy_score(s['hand'], s['distant']) * 100
            print("Agreement score \t", round(acc, 1))
        if a.get('interrater'):
            kapl12, kap12 = self._kappa(s['raters'], 'ann1', 'ann2')
            print("Kappa 1 : 2 @ ", kapl12,  "\t", round(kap12, 2))
            if not self.paper:
                kapl23, kap23 = self._kappa(s['raters'], 'ann2', 'ann3')
                kapl13, kap13 = self._kappa(s['raters'], 'ann1', 'ann3')
                print("Kappa 2 : 3 @ ", kapl23, "\t", round(kap23, 2))
                print("Kappa 1 : 3 @ ", kapl13, "\t", round(kap13, 2))


def data_to_batches(db_id, label_mapping):
    """Convert messages in db to fasttext format and filter non-english.

    Files are written to ./data/{db_id}.dataf.

    Parameters
    ----------
    db_id : str
        The string identifier of the database to be converted to fasttext
        format.

    label_mapping : dict
        A label -> int mapping to convert the labels to a number.

    Returns
    -------
    None

    """
    aff = '_fix' if 'query' in db_id or 'twitter' in db_id else ''
    db = DB(db_id + '_msg' + aff, 'r')
    ft = open('./data/' + db_id + '.dataf', 'w')

    NLP = spacy.load('en')

    user_ids, _ = reconstruct_ids(db_id)
    batch, bin_lab = [], label_mapping
    label, cur_user = str(), str()

    log("Processing " + db_id + "...")

    for line in db.loop():
        prep = "" if not cur_user else "\n"
        if cur_user != line['user_id']:
            # log("Processing " + str(line['user_id']))
            batch = []
            try:
                label = str(bin_lab[user_ids[line['user_id']]])
            except KeyError:
                # log("User error...")
                continue
            cur_user = line['user_id']
        if label:
            text = line['tweet_text'].replace('\n', ' ').replace('\t', ' ')
            batch.append(text)
            if len(batch) == 200:
                tweets = '\t'.join(batch)
                # NOTE: language detection was found to hurt performance
                # if detect(tweets) == 'en':
                tokens = ' '.join([t.text for t in NLP.tokenizer(tweets)])
                tokens = tweets
                if tokens[0] == ' ':
                    tokens = tokens[1:]
                tokens = tokens.replace('\r', ' ')
                log("Wrote tweet batch...")
                ft.write(prep + '__label__{0} '.format(label) + tokens)
                batch = []
    ft.close()


def batches_to_sets(db_id, test_size=0.2):
    """Split fasttext batches into train and test."""
    ft = open('./data/' + db_id + '.dataf', 'r')
    batches = ft.read().split('\n')
    n_test = round(len(batches) * test_size)
    n_train = len(batches) - n_test
    train = batches[:n_train]
    test = batches[-n_test:]
    with open('./data/' + db_id + '.train', 'w') as ftrain:
        [ftrain.write(x + '\n') for x in train]
    with open('./data/' + db_id + '.test', 'w') as ftest:
        [ftest.write(x + '\n') for x in test]

if __name__ == "__main__":
    ans = AnnotationStats(paper=True, labels=True, bots=True, agreement=True,
                          interrater=True)
    ans.report()

    lm = {'m': 0, 'f': 1, 'M': 0, 'F': 1}
    # data_to_batches(db_id='twitter_gender', label_mapping=lm)
    data_to_batches(db_id='query_gender', label_mapping=lm)
    data_to_batches(db_id='plank_gender', label_mapping=lm)
    data_to_batches(db_id='volkova_gender', label_mapping=lm)

    # batches_to_sets(db_id='twitter_gender', test_size=0.2)
    batches_to_sets(db_id='query_gender', test_size=0.2)
    batches_to_sets(db_id='plank_gender', test_size=0.2)
    batches_to_sets(db_id='volkova_gender', test_size=0.2)
