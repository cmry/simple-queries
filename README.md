Repository for the work described in [Simple Queries as Distant Labels for
Predicting Gender on Twitter](http://noisy-text.github.io/2017/pdf/WNUT07.pdf),
presented at [W-NUT 2017](http://noisy-text.github.io/2017/index.html).

The code is released under the MIT license, the data under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/).

If you use anything in our repository, please cite the following work:

```
@inproceedings{emmery2017simple,
  title={Simple Queries as Distant Labels for Predicting Gender on Twitter},
  author={Emmery, Chris and Chrupa\l{}a, Grzegorz and Daelemans, Walter},
  booktitle={3d Workshop on Noisy User-generated Text (W-NUT 2017)},
  year={2017}
}
```

Also consider citing the corpora that we used to compare when using the readers
supplied with our work. See the [docs]() for references.

## Overview

- [tl;dr]()
- [Quick Start]()
  - [Reproduction]()
  - [Build your Own Classifier]()
- [Dependencies]()
- [Paper Data]()

## tl;dr

This repository offers scripts to automatically label Twitter users by gender,
using simple queries for self-reports ("*I'm a {girl, boy, man, woman, ...}*").
Using batches of 200 tweets from their timelines, we train a
[fastText](https://github.com/facebookresearch/fastText) model to classify
**other users** by gender. This distant supervision achieves
comparable, or better performance than state-of-the-art methods, without the
need of manual annotation. The collection can be repeated weekly and yields
roughly 10k profiles.


## Quick Start

Not only do we supply code to *replicate* our experiments, we also offer a
flexible API to use the same method for distant labeling of users by gender.

> **Support Disclaimer:** The code is written for MacOS and Linux systems. This
  limitation is partly due to fastText.

### Reproduction

Simply run the `sec...py` files in this order: data -> proc -> exp -> res:

```shell
python3 sec3_data.py
python3 sec3_proc.py
sh sec4_exp.sh
python3 sec5_res.py
```

> **Important Note**: Please make sure that your own Twitter API keys are added
  in `misc_keys` and the correct files are in `corpora` if you'd like to
  replicate the comparison with the other corpora. Only our own data comes with
  the repository!

If you don't want to run the comparisons, comment out the `Plank` and `Volkova`
parts for every `__main__` boilerplate at the end of each `.py` file (not the
`.sh`). If you'd like to add **newly** collected data in for the comparison,
uncomment the `twitter` parts in the same boilerplate.

> **Note on Score Reproduction**: The **annotation** scores differ slightly,
  as the exact user ids that were annotated when writing the paper differ from
  the complete set (the reproduction shows an almost exact approximation). With
  regards to the **classification** scores: i) Twitter's Terms of Service
  formally prohibit sharing the message IDs, and ii) a large amount of
  accounts and tweets have been removed since the date of collection.

### Build your Own Gender Classifier

You can supply a general text wrapping in `query_string`. We assumed that
people self-report using "*I'm a {girl, boy, man, woman}*" etc.

```python
query_string = 'm a {0}'
query_words = {'girl': 'f', 'boy': 'm', 'man': 'm', 'woman': 'f', 'guy': 'm',
               'dude': 'm', 'gal': 'f', 'female': 'f', 'male': 'm'}
```

There's a list for *removing* tweets (`filters`) by certain patterns like
retweets, or quotes (e.g. 'Random woman: "I'm a man"'), that might not be
self-reports. Moreover, the gender label can be flipped if certain words
or in the full tweet (`flip_any`) or *before* the query string (`flip_prefix`).
Examples where this might be relevant are for example "don't assume
I'm a girl".

> Please note that while all examples focus on a binary representation of
  gender (as this is common in related work), the code to a certain extent also
  allows for the inclusion of other gender identities by altering the
  `query_words` mapping. In this case however, the label flipping logically does
  not work.

```python
fil = ['rt ', '"', ': ']
flp_any = ["according to", "deep down"]
flp_pfx = [" feel like ", " where ", " as if ", " hoping ", " assume ",
           " think ", " assumes ", " assumed ", " assume that ",
           " assumed that ", " then ", " expect that ", " expect ",
           "that means ", " means ", " think ", " implying ", " guess ",
           " thinks ", " tells me ", " learned ", " if "]
```

#### Collecting Data

From here, collection is simply setting all this information in the
`DistantCollection` class initialization, and running the collection methods.

> Make sure you've added your own Twitter API keys to `misc_keys` before running!

```python
from sec3_data import DistantCollection

dc = DistantCollection(query_string=qs, query_words=qw, filters=fil,
                       flip_any=flp_any, flip_prefix=flp_pfx,
                       clean_level='messages', db_id='your_db_id')
```

And finally fetching the users, and their timelines:

```python
dc.fetch_query_tweets()  # users
dc.fetch_user_tweets()   # timelines
```

#### Preprocessing and Batching

Once the collection is done, you need to set up a label mapping (depending) on
what you've entered as keys in `query_words`, and run that through the
preprocessing function:

```python
from sec3_proc import data_to_batches

lm = {'m': 0, 'f': 1, 'M': 0, 'F': 1}
data_to_batches(db_id='your_db_id', label_mapping=lm)
```

#### Scoring

Using the `fastText` function prints a train / test evaluation. If you'd like
to run `MajorityBaseline` and the `LexiconGender` classifier from Sap et al.,
use the code below:

```python
from sec5_res import MajorityBaseline, LexiconGender, fastText

bl = MajorityBaseline()
lg = LexiconGender()

train = test = 'your_db_id'
fastText(train + '_gender', test + '_gender')  # outputs results

fin = open('./data/{0}_gender.test'.format(test)).readlines()
sscore = round(lg.lex_score(fin), 3)
bscore = round(bl.mb_score(fin), 3)
print("Sap baseline @ test: {}".format(sscore))
print("Maj baseline @ test: {}".format(bscore))
```

This should cover setting up your own classifier. If you'd like to re-use
fastText after training is done, refer to their documentation and change
`sec4_exp.sh` accordingly.

## Dependencies

These are the packages (and their version tested with the latest version of
the repository):

```
pandas        0.20.1
tweepy        3.5.0
langdetect    1.0.7 (optional)
scikit-learn  0.18.1
spacy         1.9.0
```

## Paper Data

The data can be found under `./corpora/query-gender.json`. The file is
structured as follows:

```json
{
    "annotations": {
        "user_id": {
            "ann1": "m / f / o / - / 0",
            "ann2": "m / f / o / - / 0",
            "ann3": "m / f / o / - / 0",
            "bot": "True / False",
            "majority": "m / f",
            "query_label": "m / f",
            "query_label2": "m / f"
        },
        "...": {
        },
    },
    "filtered": {
        "tweet_id": "user_id",
        "...": ,
    }
}
```

Not all three annotators annotated every profile. The annotations from the
paper only used 2 annotators; an approximate implementation of this evaluation
is provided in `sec5_res.py`. Annotation field values are: `m` for male, `f`
for female, `o` anything else, `-` not sure, and `0` for not annotated. A
majority decision based on the three annotators is provided in `majority` (only
based on `m` / `f` annotations). There's also a label if the user was
considered to be a bot (this label wasn't used in the paper), and there's two
tags (`query_label`) provided by the Query heuristic from the paper. The
first is the label **without** using the rule to flip the label, the second
**with** the label flipped if necessary. Finally, the original Query hit
tweets (that were removed from the final train and test data) are listed under
`filtered`. 
