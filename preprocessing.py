import re
from itertools import cycle
import tensorflow.keras.utils as ku
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from random import sample
import json


def clean(text):
    s = text.lower()
    s = re.sub(r'\n', ' <PARAGRAPHEND> ', s)
    s = re.sub(r'\[\d+\]', ' ', s)
    s = re.sub(r"\d+", ' <NUM> ', s)
    s = re.sub(r"\d+.\d+", ' <NUM> ', s)
    s = re.sub(r"\(.*?\)", ' ', s)
    s = s.replace(':', ' <SENTENCEEND> ')
    s = s.replace('։', ' <SENTENCEEND> ')
    s = s.replace('...', ' <ELIPSIS> ')
    s = s.replace('՞', '')
    s = s.replace(r'%', ' տոկոս')
    s = s.replace('՛', '')
    s = s.replace('՜', '')
    s = s.replace('՝', '')
    s = s.replace(',', '')
    s = s.replace('-', ' ')
    s = s.replace('', '')
    s = s.replace('«', '')
    s = s.replace('»', '')
    s = re.sub(r'\s+', ' ', s)
    s = s.replace('<NUM> <NUM>', '<NUM>')
    s = s.replace('<NUM><NUM>', '<NUM>')
    return s


def load_data(filename):
    rows = []
    with open(filename, 'r', encoding="utf8") as file:
        for value in file:
            rows.append(value[:-1])
    return rows


def create_tokenizer(corpus, vocab_size=80000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(corpus)

    tokenizer_json = tokenizer.to_json()

    with open('tokenizer.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        print('Tokenizer saved.')


def save_numeric_data(
        filename='data_final/2sentences.txt',
        N=1613824,
        files_dir='data_final',
        max_sequence_len=80,
        batch_size=10240,
        oov_thresh=0.2
):
    data = []
    with open(filename, 'r', encoding="utf8") as file:
        for value in file:
            data.append(value[:-1])
    print('data loaded')
    corpus_train, corpus_test = train_test_split(data, train_size=0.95, random_state=5)
    corpus_train = sample(corpus_train, N)

    # checking for existing tokenizer file
    if 'tokenizer.json' not in os.listdir():
        # creating tokenizer file
        create_tokenizer(corpus=corpus_train, vocab_size=80000)
        print('tokenizer.json created')
    else:
        # using already existing tokenizer
        with open('tokenizer.json') as f:
            totenizer_data = json.load(f)
            tokenizer = tokenizer_from_json(totenizer_data)
        print('using existing tokenizer')

    # train set
    # creating n-gram sequences
    input_sequences = []
    for line in corpus_train:
        token_list = tokenizer.texts_to_sequences([line])[0]
        # drop sequences with more than max_oov unknown words
        if token_list.count(1)/len(token_list) > oov_thresh:
            continue
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    print(len(input_sequences), 'n-gram sequences created')
    # padding sequences and saving by batches
    np.random.shuffle(input_sequences)
    batch = []
    file_id = 0
    for i in input_sequences:
        batch.append(i)
        if len(batch) == batch_size:
            batch = np.array(pad_sequences(batch, maxlen=max_sequence_len, padding='pre'))
            np.save(files_dir+'/train/train'+str(file_id)+'.npy', batch)
            print('file train'+str(file_id)+'.npy saved')
            file_id = file_id+1
            batch = []

    # test set
    # creating n-gram sequences
    input_sequences = []
    for line in corpus_test:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
    print(len(input_sequences), 'n-gram sequences created')
    # padding sequences and saving by batches
    np.random.shuffle(input_sequences)
    batch = []
    file_id = 0
    for i in input_sequences:
        batch.append(i)
        if len(batch) == batch_size:
            batch = np.array(pad_sequences(batch, maxlen=max_sequence_len, padding='pre'))
            np.save(files_dir+'/test/test'+str(file_id)+'.npy', batch)
            print('file test'+str(file_id)+'.npy saved')
            file_id = file_id+1
            batch = []


def batch_generator(files_dir, n_files=1000, batch_size=512, vocab_size=80000):
    files = sample(os.listdir(files_dir), n_files)
    files = cycle(files)
    for f in cycle(files):
        data = np.load(files_dir+f)
        for n in range(int(data.shape[0]/batch_size)):
            x_batch = data[n*batch_size:(n+1)*batch_size, :-1]
            y_batch = data[n*batch_size:(n+1)*batch_size, -1]
            y_batch = ku.to_categorical(y_batch, num_classes=vocab_size + 1, dtype='int32')
            yield (x_batch, y_batch)
