from numpy import array, asarray, zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
import keras
from utils import tokenize

INPUT_DIM = 300
NUM_CLASSES = 6


def get_docs(filenames):
    docs = []
    labels = []
    max_length = 0
    class_id = 0
    for filename in filenames:
        with open(filename) as f:
            actual_lines = []
            for line in f:
                tokenized = tokenize(line)
                actual_size = len(tokenized)
                if actual_size > max_length:
                    max_length = actual_size
                actual_lines += ' '.join(tokenized)
            labels += [class_id] * len(actual_lines)
            docs += actual_lines
        class_id += 1
    return docs, labels, max_length


# define documents
docs, labels, max_length = get_docs(['alegria.txt', 'asco.txt', 'ira.txt',
                 'miedo.txt', 'sorpresa.txt', 'tristeza.txt'])

# define class labels
# labels = array(labels)
labels = keras.utils.np_utils.to_categorical(labels, NUM_CLASSES)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
# print(encoded_docs)
# pad documents to a max length of 4 words
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('model.w2v.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, INPUT_DIM))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
model.add(Embedding(vocab_size,
                    INPUT_DIM,
                    weights=[embedding_matrix],
                    input_length=max_length,
                    trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(6, activation='softmax'))

# compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
