import re
import gensim
import preprocessor as p


def tokenize(text):
    """
    Applies basic tweet-processing, such as removing users, retweets, hashtags,
    etc. then tokenizes the string.
    Ignored indexes of a preprocessed string:
        0 -> ID of the tweet.
        n-1 and n -> timestamp of the tweet.
    :param text: Text to be parsed.
    :return: Text parsed.
    """
    return gensim.utils.simple_preprocess(p.clean(text))


def remove_dates(tokens):
    parsed = []
    for token in tokens:
        if not re.match('^(?:(?:[0-9]{2,4}[-:,]){2}[0-9]{2}|am|pm)$', token):
            parsed.append(token)

    return str(parsed)


def simple_preprocessing(text):
    return p.clean(text)
