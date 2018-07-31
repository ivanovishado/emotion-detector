import gzip
import gensim
import logging
import preprocessor as p

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MySentences:
    def __init__(self, filename):
        self.filename = filename
        self.len = 0
        self.len_set = False

    def __iter__(self):
        logging.info(
            "reading file {0}...this may take a while".format(self.filename))
#        with gzip.open(self.filename, 'rb') as f:
        with open(self.filename, encoding='utf-8') as f:
            for line in f:
                if not self.len_set:
                    self.len += 1
#                yield gensim.utils.simple_preprocess(line)
                yield gensim.utils.simple_preprocess(p.clean(line))

        self.len_set = True

    def __len__(self):
        return self.len


documents = MySentences('tweets.txt')
model = gensim.models.Word2Vec(
    documents,
    size=300,
    window=10,
    min_count=2,
    workers=10)
model.train(documents, total_examples=len(documents), epochs=10)

model.wv.save_word2vec_format('model.w2v.txt')
