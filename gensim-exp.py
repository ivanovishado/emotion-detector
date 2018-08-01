import re
import gensim
import logging
import preprocessor as p

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def process_text(text):
    """
    Applies basic tweet-processing, such as removing users, retweets, hashtags,
    etc. then tokenizes the string.
    Ignored indexes of a preprocessed string:
        0 -> ID of the tweet.
        n-1 and n -> timestamp of the tweet.
    :param text: Text to be parsed.
    :return: Text parsed.
    """
    return gensim.utils.simple_preprocess(p.clean(text))[1:-2]


def remove_dates(tokens):
    parsed = []
    for token in tokens:
        if not re.match('^(?:(?:[0-9]{2,4}[-:,]){2}[0-9]{2}|am|pm)$', token):
            parsed.append(token)

    return parsed


class MySentences:
    def __init__(self, filename):
        self.filename = filename
        self.len = 0
        self.len_set = False

    def __iter__(self):
        logging.info(
            "reading file {0}...this may take a while".format(self.filename))
        with open(self.filename, encoding='utf-8') as f:
            for line in f:
                if not self.len_set:
                    self.len += 1
                yield process_text(line)

        self.len_set = True

    def __len__(self):
        return self.len


documents = MySentences('corpus.txt')
model = gensim.models.Word2Vec(
    documents,
    size=300,
    window=10,
    min_count=2,
    workers=10)
model.train(documents, total_examples=len(documents), epochs=10)

model.wv.save_word2vec_format('model.w2v.txt')
