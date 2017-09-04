"""Scripts to run the Data Collection part of the paper."""

import json
from misc_keys import twitter_keys
import pandas as pd
from time import localtime
from time import sleep
from time import strftime
import tweepy


def log(message):
    """Print simple timestamped message log."""
    entry = "{0} - {1}".format(strftime('%H:%M:%S', localtime()), message)
    print(entry)
    return entry


def chunk(l, n):
    """Divide list l into n chuncks."""
    n = int(len(l) / n)
    for i in range(0, len(l), n):
        yield l[i:i + n]


def tw_connect(keys):
    """Connect to Twitter API using Tweepy.

    Parameters
    ----------
    keys : dict
        Dictionary object containing keys app_public, app_secret, per_public,
        and per_secret.

    Returns
    -------
    api : object
        Authenticated Tweepy API object.

    """
    auth = tweepy.OAuthHandler(keys['app_public'], keys['app_secret'])
    auth.set_access_token(keys['per_public'], keys['per_secret'])
    return tweepy.API(auth)

try:
    assert twitter_keys['app_public']
    API = tw_connect(twitter_keys)
except AssertionError:
    log("API keys are empty. Please provide them in misc_keys.py...")
    exit()


class DB(object):
    """Super simple database class.

    Parameters
    ----------
    fdir : str
        File directory.

    mode : str
        File mode ('r' for read, 'w' for write).

    Attributes
    ----------
    db : obj
        File object.

    """

    def __init__(self, db_name, mode):
        """Open file directory."""
        self.mode = mode
        try:
            self.db = open('./data/' + db_name + '.db', mode)
        except FileNotFoundError:
            fo = open('./data/' + db_name + '.db', 'w')
            fo.close()
            self.db = open('./data/' + db_name + '.db', mode)

    def insert(self, jsonf):
        """Write json line to file."""
        self.db.write(json.dumps(jsonf) + "\n")

    def commit(self):
        """Write changes to disk."""
        self.db.close()

    def fetch_key(self, key):
        """Fetch values for key."""
        for jsf in self.loop():
            yield jsf[key]

    def loop(self):
        """Iterate through db."""
        assert self.mode == 'r'
        for line in self.db:
            jsf = json.loads(line)
            yield jsf


def reconstruct_ids(db_id):
    """Extract query and user information from existing file."""
    uds = DB(db_id + '_fix', 'r')
    user_ids, query_ids = {}, {}
    if not uds.db.read():
        uds = DB(db_id, 'r')
    for line in uds.loop():
        try:
            user_ids[line['user_id']] = line['label']
            query_ids[line['tweet_id']] = line['query']
        except KeyError:
            user_ids[line['id']] = line['label']
    return user_ids, query_ids


class DistantCollection(object):
    """Reader class to collect, store, and access the distant gender set.

    Parameters
    ----------
    db_id : str
        String identifier for this specific database (so the name).

    query_string : str
        Should contain the wrapper string where the keywords pairs should be
        inserted. As such it should be of the format 'this is the query and
        use {0} for the word position'. So, for '"I'm a girl"', where girl is
        variable, you write '"I'm a {0}"'. Note that " can be used for exact
        matches.

    query_words : dict
        Dictionary where the keys should be the query words inserted in the
        query_string, and the values the labels associated with these words.

    filter : list
        Patterns that should be completely ignored as they do not guarantee a
        message to be a self-report (retweets for example).

    flip_any : list
        Substrings that could flip the gender label, and can be positioned
        anywhere in a tweet (e.g. " according to " such and such
        " I'm a girl ").

    flip_prefix : list
        Substrings that could flip the gender label, and can be positioned
        only as a suffix to the query (e.g. I " guess "" I'm a man " now).

    clean_level : str, {'historic', 'query'}, default: 'historic'
        The level on which to remove the query strings. This can be either from
        history, meaning it will remove any of the queries from the entire
        timeline (like in the paper), or only on query level, meaning it will
        remove the queries only.

    mode : str, {'live', 'test'}, default: 'live'
        If set to 'test', will only run a few iterations of data collection
        (ideal for debugging and such).

    Attributes
    ----------
    id : str
        String identifier for this specific database.

    hits : obj
        Database wrapper for the query table. Includes the orginal query
        hit message, and a user and tweet id.

    users : obj
        Database wrapper for user table. Normal query representation doesn't
        include these objects!

    messages : obj
        Database wrapper for message table.

    hit_fix : obj
        Database wrapper with the corrected distant labels initially provided
        by the naive queries. These are either flipped or filtered.

    msg_fix : obj
        Database wrapper for the corrected message table. This excludes tweets
        that include any of the specified queries.

    queries : dict
        Formatted dictionary combining query_string and query_words so that
        {full_query: distant_label}.

    filter : list
        See filters parameter.

    flip_any : list
        See flip_any parameter.

    flip_prefix : list
        See flip_prefix parameter.

    user_ids : dict
        Dictionary so that {user_id : label}.

    query_ids : dict
        Dictionary so that {message id containing query: query}.

    clean_level : str
        See clean_level parameter.

    max : int
        0 if mode == 'live' else 1

    """

    def __init__(self, query_string, query_words, filters, flip_any,
                 flip_prefix, clean_level='messages', mode='live',
                 db_id='twitter_gender'):
        """Set collection and queries."""
        self.id = db_id
        self.hits = DB(self.id, 'a')
        self.users = DB(self.id + '_usr', 'a')
        self.messages = DB(self.id + '_msg', 'a')
        self.hit_fix = DB(self.id + '_fix', 'a')
        self.msg_fix = DB(self.id + '_msg_fix', 'a')

        self.queries = {query_string.format(k): v for
                        k, v in query_words.items()}
        self.filter = filters
        self.flip_any = flip_any
        self.flip_prefix = flip_prefix

        self.user_ids = dict()
        self.query_ids = dict()

        self.clean_level = clean_level
        self.max = 0 if mode == 'live' else 1

    def remove_query_tweets(self):
        """Remove query hits from tweets."""
        db = DB(self.id + '_msg', 'r')
        for line in db.loop():
            label = self.user_ids[line['user_id']]
            line['distant_label'] = label
            if self.clean_level == 'messages':
                if not any([query in line['tweet_text'].lower()
                            for query in self.queries]):
                    self.msg_fix.insert(line)
            else:
                if line['tweet_id'] not in self.query_ids:
                    self.msg_fix.insert(line)

    def flip_label(self, uid, tid, text):
        """Return flipped label if rules in text, return none if in filter."""
        query_tails = [" ", ".", "!", ",", ":", ";"]  # etc
        if any([it in text for it in self.filter]):  # if illegal
            return
        label = self.user_ids[uid]
        query = self.query_ids[tid]
        if any([query + affix in text for affix in query_tails]):
            if any([f in text for f in self.flip_any]):
                label = 'm' if label == 'f' else 'f'
            elif any([p + query in text for p in self.flip_prefix]):
                label = 'm' if label == 'f' else 'f'
        return label

    def correct_query_tweets(self):
        """Correct the query tweets using heuristics, write to new file."""
        db = DB(self.id, 'r')
        for line in db.loop():
            new_label = self.flip_label(line['user_id'],
                                        line['tweet_id'],
                                        line['tweet_text'].lower())
            if new_label:
                line['label'] = new_label
                self.hit_fix.insert(line)
        self.hit_fix.commit()

    def get_users(self, cursor, label, query):
        """Given a query cursor, store user profile and label."""
        try:
            for page in cursor.pages():
                log("flipping page...")
                for tweet in page:
                    try:
                        self.user_ids[tweet.user.id] = label
                        self.query_ids[tweet.id] = query
                        self.users.insert(tweet.user._json)
                        self.hits.insert({'user_id': tweet.user.id,
                                          'tweet_id': tweet.id,
                                          'tweet_text': tweet.text,
                                          'label': label,
                                          'query': query})
                    except Exception as e:
                        log("error getting users: " + str(e))
                if self.max:
                    break

        except tweepy.TweepError:
            log("Rate limit hit, going to zzz....")
            sleep(300)
            self.get_users(cursor, label, query)

    def get_queries(self):
        """Search Twitter API for tweets matching queries and fetch users."""
        for query, label in self.queries.items():
            query = '"' + query + '"'
            cursor = tweepy.Cursor(API.search, q=query, include_entities=True,
                                   count=200)
            self.get_users(cursor, label, query)
            if self.max:
                break

    def fetch_query_tweets(self):
        """Given query assignments, collect query tweets with API tokens."""
        self.get_queries()
        self.hits.commit()
        self.users.commit()
        self.correct_query_tweets()

    def get_tweets(self, cursor):
        """Given a timeline cursor, fetch tweets and remove user object."""
        try:
            for page in cursor.pages():
                for tweet in page:
                    tweet = tweet._json
                    del tweet['user']
                    yield tweet
        except tweepy.TweepError:
            log("Rate limit hit, going to zzz....")
            sleep(5)
            self.get_tweets(cursor)

    def get_timelines(self):
        """Given ID assignments, collect timelines with provided API tokens."""
        for user_id in self.user_ids:
            cursor = tweepy.Cursor(API.user_timeline, id=user_id, count=200)
            for tweet in self.get_tweets(cursor):
                tweet['user_id'] = user_id
                assert not tweet.get('user')
                self.messages.insert({'tweet_id': tweet['id'],
                                      'user_id': user_id,
                                      'tweet_text': tweet['text']})
            log("Fetched user...")
            if self.max:
                break

    def fetch_user_tweets(self):
        """Divide all ids amongst API connections and thread them."""
        try:
            assert self.user_ids
        except (AssertionError, AttributeError):
            log("Empty users, trying to repopulate from " + self.id +
                "_fix.db. If this errors, make sure you ran " +
                "fetch_query_tweets first!")
            self.user_ids, self.query_ids = reconstruct_ids(self.id)

        self.get_timelines()
        self.messages.commit()
        if self.id == 'twitter_gender' or self.id == 'query_gender':
            self.remove_query_tweets()


class QueryCollection(DistantCollection):
    r"""Reader class to load and store our Query corpus.

    Parameters
    ----------
    db_id : str
        String identifier for this specific database (so the name).

    corpus_dir : str, optional, default ./corpora/query-gender.json
        Directory where corpus is located.

    clean_level : str, {'historic', 'query'}, default: 'historic'
        The level on which to remove the query strings. This can be either from
        history, meaning it will remove any of the queries from the entire
        timeline (like in the paper), or only on query level, meaning it will
        remove the queries only.

    mode : str, {'live', 'test'}, default: 'live'
        If set to 'test', will only run a few iterations of data collection
        (ideal for debugging and such).

    Attributes
    ----------
    id : str
        String identifier for this specific database.

    users : obj
        Database wrapper for user table.

    messages : obj
        Database wrapper for message table.

    msg_fix : obj
        Database wrapper for the corrected message table. This excludes tweets
        that include any of the specified queries.

    queries : dict
        Formatted dictionary combining query_string and query_words from the
        paper so that {full_query: distant_label}.

    user_ids : dict
        Dictionary so that {user_id : label}.

    clean_level : str
        See clean_level parameter.

    max : int
        0 if mode == 'live' else 1

    corpus : dict
        JSON object with Query corpus.

    Notes
    -----
    The Query corpus is from our own paper:

    @article{emmery2017simple,
      title={Simple Queries as Distant Labels for Predicting Gender on Twitter
             },
      author={Chris Emmery, Grzegorz Chrupa{\l}a, Walter Daelemans},
      journal={WNUT 2017},
      pages={50-55},
      year={2017}
    }

    """

    def __init__(self, db_id='query_gender',
                 corpus_dir='./corpora/query-gender.json',
                 clean_level='messages', mode='live'):
        """Call correct databases corresponding to class, open corpus."""
        self.id = db_id
        self.users = DB(self.id, 'a')
        self.messages = DB(self.id + '_msg', 'a')
        self.msg_fix = DB(self.id + '_msg_fix', 'a')

        # NOTE: these are hardcoded to reproduce the paper
        query_string = 'm a {0}'
        query_words = {'girl': 'f', 'boy': 'm', 'man': 'm', 'woman': 'f',
                       'guy': 'm', 'dude': 'm', 'gal': 'f', 'female': 'f',
                       'male': 'm'}
        self.queries = {query_string.format(k): v for
                        k, v in query_words.items()}

        self.user_ids = dict()

        self.clean_level = clean_level
        self.max = 0 if mode == 'live' else 1

        try:
            self.corpus = json.load(open(corpus_dir, 'r'))
        except FileNotFoundError:
            log("Something went wrong while loading the query corpus. " +
                "Re-download from http://github.com/cmry/simple-queries " +
                "and store in ./corpora")

    def fetch_users(self):
        """Collect the users in the Query corpus."""
        userd = {}
        for idx, info in self.corpus['annotations'].items():
            userd[idx] = info['query_label2']
            if len(userd) == 100:
                log("Getting user batch...")
                userl = list(userd.keys())
                users = API.lookup_users(userl)
                for user in users:
                    user = user._json
                    user['label'] = userd[user['id_str']]
                    self.users.insert(user)
                userd = {}
                if self.max:
                    break
        self.users.commit()


class PlankCollection(DistantCollection):
    """Reader class to load and store English part of TwiSty corpus.

    Parameters
    ----------
    db_id : str
        String identifier for this specific database (so the name).

    corpus_dir : str, optional, default ./corpora/TwiSty-EN.json
        Directory where corpus is located.

    mode : str, {'live', 'test'}, default: 'live'
        If set to 'test', will only run a few iterations of data collection
        (ideal for debugging and such).

    Attributes
    ----------
    id : str
        String identifier for this specific database.

    users : obj
        Database wrapper for user table.

    messages : obj
        Database wrapper for message table.

    user_ids : dict
        Dictionary so that {user_id : label}.

    max : int
        0 if mode == 'live' else 1

    corpus : dict
        JSON object with English part of the TwiSty corpus.

    Notes
    -----
    The English part of the TwiSty corpus is orignally from:

    @inproceedings{plank-hovy:2015,
      author={Barbara Plank and Dirk Hovy},
      title={Personality Traits on Twitter---Or---How to Get 1,500 Personality
             Tests in a Week}
      booktitle={The 6th Workshop on Computational Approaches to Subjectivity,
                 Sentiment and Social Media Analysis (WASSA), EMNLP 2015.}
      year=2015,
    }

    """

    def __init__(self, db_id='plank_gender',
                 corpus_dir='./corpora/TwiSty-EN.json', mode='live'):
        """Call correct databases corresponding to class, open corpus."""
        self.id = db_id
        self.users = DB(self.id, 'a')
        self.messages = DB(self.id + '_msg', 'a')

        self.user_ids = dict()

        self.max = 0 if mode == 'live' else 1

        try:
            self.corpus = json.load(open(corpus_dir, 'r'))
        except FileNotFoundError:
            log("Please request TwiSty-EN from ",
                "http://www.clips.ua.ac.be/datasets/twisty-corpus ",
                "and store in ./corpora")

    def fetch_users(self):
        """Collect the users in the Plank corpus."""
        userd = {}
        for idx, info in self.corpus.items():
            userd[info['user_id']] = info['gender']
            if len(userd) == 100:
                log("Getting user batch...")
                userl = list(userd.keys())
                users = API.lookup_users(userl)
                for user in users:
                    user = user._json
                    user['label'] = userd[user['id_str']]
                    self.users.insert(user)
                userd = {}
        self.users.commit()


class VolkovaCollection(DistantCollection):
    """Reader class to load and store the corpus from Volkov et al.

    Parameters
    ----------
    db_id : str
        String identifier for this specific database (so the name).

    corpus_dir : str, optional, default ./corpora/userIDToAttributes
        Directory where corpus is located.

    mode : str, {'live', 'test'}, default: 'live'
        If set to 'test', will only run a few iterations of data collection
        (ideal for debugging and such).

    Attributes
    ----------
    id : str
        String identifier for this specific database.

    users : obj
        Database wrapper for user table.

    messages : obj
        Database wrapper for message table.

    user_ids : dict
        Dictionary so that {user_id : label}.

    max : int
        0 if mode == 'live' else 1

    corpus : dict
        CSV file with English part of the TwiSty corpus.

    Notes
    -----
    The Volkova corpus is from:

    @article{Volkova:16Interest,
      title={Mining User Interests to Predict Perceived Psycho-Demographic
             Traits on Twitter},
      author={Volkova, Svitlana and Bachrach, Yoram and Van Durme, Benjamin},
      journal={Proceddings of the 2nd IEEE International Conference On Big Data
               Computing Service And Applications (IEEE BigData 2016)},
      year={2016}
    }

    """

    def __init__(self, db_id='volkova_gender',
                 corpus_dir='./corpora/userIDToAttributes', mode='live'):
        """Call correct databases corresponding to class, open corpus."""
        self.id = db_id
        self.users = DB(self.id, 'a')
        self.messages = DB(self.id + '_msg', 'a')
        self.user_ids = dict()
        self.max = 0 if mode == 'live' else 1

        try:
            with open(corpus_dir, 'r') as fi:
                with open(corpus_dir + '_f', 'w') as fo:
                    fs = fi.read()
                    fo.write(fs.replace('::', ''))
            with open(corpus_dir + '_f') as new_fi:
                self.corpus = pd.DataFrame.from_csv(new_fi, sep='\t')
        except FileNotFoundError:
            log("Please request acces to ",
                "https://bitbucket.org/svolkova/psycho-demographics ",
                "from Svitlana Volkova (http://www.cs.jhu.edu/~svitlana/) "
                "and store the userIDToAttributes file in ./corpora")

    def fetch_users(self):
        """Collect the users in the Volkova corpus."""
        userd = {}
        for _id, cols in self.corpus.iterrows():
            userd[_id] = 'f' if cols['gender'] == 'Female' else 'm'
            if len(userd) == 100:
                log("Getting user batch...")
                userl = list(userd.keys())
                users = API.lookup_users(userl)
                for user in users:
                    user = user._json
                    user['label'] = userd[user['id_str']]
                    self.users.insert(user)
                userd = {}
                if self.max:
                    break

if __name__ == "__main__":

    qs = 'm a {0}'
    qw = {'girl': 'f', 'boy': 'm', 'man': 'm', 'woman': 'f', 'guy': 'm',
          'dude': 'm', 'gal': 'f', 'female': 'f', 'male': 'm'}

    fil = ['rt ', '"', ': ']
    flp_any = ["according to", "deep down"]
    flp_pfx = [" feel like ", " where ", " as if ", " hoping ", " assumed ",
               " think ", " assumes ", " assumed ", " assume that ",
               " assumed that ", " then ", " expect that ", " expect ",
               "that means ", " means ", " think ", " implying ", " guess ",
               " thinks ", " tells me ", " learned ", " if "]

    dc = DistantCollection(query_string=qs, query_words=qw, filters=fil,
                           flip_any=flp_any, flip_prefix=flp_pfx,
                           clean_level='messages', db_id='twitter_gender')

    # log("Fetching query tweets...")
    # dc.fetch_query_tweets()
    # log("Fetching user tweets...")
    # dc.fetch_user_tweets()

    log("Fetching Query users...")
    tc = QueryCollection(db_id='query_gender', clean_level='messages')
    tc.fetch_users()
    log("Fetching Query tweets...")
    tc.fetch_user_tweets()

    log("Fetching Plank users...")
    tc = PlankCollection(db_id='plank_gender')
    tc.fetch_users()
    log("Fetching Plank tweets...")
    tc.fetch_user_tweets()

    log("Fetching Volkova users...")
    pc = VolkovaCollection(db_id='volkova_gender')
    pc.fetch_users()
    log("Fetching Volkova tweets...")
    pc.fetch_user_tweets()
