import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(BASE_DIR, 'train')

labels = []
texts = []

for label_type in ['neg','pos']:
    dirname = os.path.join(train_dir, label_type)
    for filename in os.listdir(dirname):
        if filename[-4:] == '.txt':
            f = open(os.path.join(dirname, filename), encoding='utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

maxlen = 100
training_samples = 200
validation_samples = 10000

#Break-down works into tokens
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

# print(len(tokenizer.word_index))
# print(len(sequences))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

# print(data.shape)
indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

features_train = data[:training_samples]
labels_train = labels[:training_samples]
validation_features = data[training_samples: validation_samples + training_samples]
validation_labels = labels[training_samples: validation_samples + training_samples]

model = Sequential([
    Embedding(1000, 8, input_length=maxlen),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

#model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(features_train, labels_train, epochs=5, validation_data=(validation_features, validation_labels), batch_size=32)

print(model.evaluate(features_train, labels_train))

#model.save('imdb_model.h5')

print(model.predict(features_train)[0]) #If probability is greater than 0.6, its a positive movie review

print(labels_train[0])
