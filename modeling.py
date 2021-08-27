from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from preprocessing import clean
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku


def lstm_model(
        vocab_size,
        max_sequence_len,
        D_embed=200,
        D_lstm1=500,
        D_lstm2=300
):
    total_words = vocab_size + 1
    clear_session()
    model = Sequential()
    model.add(Embedding(total_words, D_embed, input_length=max_sequence_len - 1))
    model.add(LSTM(D_lstm1, dropout=0.5, return_sequences=True))
    model.add(Bidirectional(LSTM(D_lstm2, dropout=0.3)))
    model.add(Dense(vocab_size + 1, activation='softmax'))
    return model


def plot_metrics(filename):
    metrics = pd.read_csv(filename)
    metrics['accuracy'] = metrics['accuracy'] * 100
    metrics['val_accuracy'] = metrics['val_accuracy'] * 100
    metrics.index = metrics.index + 1
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 3.5))
    axs[0].plot(metrics[['loss', 'val_loss']])
    axs[0].set_title('Loss')
    axs[0].set(xlabel='Epochs')
    axs[0].legend(['Training', 'Validation'])
    axs[1].plot(metrics[['accuracy', 'val_accuracy']])
    axs[1].set_title('Accuracy %')
    axs[1].set(xlabel='Epochs')
    axs[1].legend(['Training', 'Validation'])
    plt.show()


def save_history(history, filename='history.csv'):
    hist_df = pd.DataFrame(
        {'accuracy': history.history['accuracy'],
         'loss': history.history['loss'],
         'val_accuracy': history.history['val_accuracy'],
         'val_loss': history.history['val_loss']}
    )
    if filename in os.listdir():
        hist_df_old = pd.read_csv(filename)[['accuracy', 'loss', 'val_accuracy', 'val_loss']]
        hist_df = hist_df_old.append(hist_df, ignore_index=True)

    hist_df.to_csv(filename)


with open('tokenizer.json') as f:
    tokenizer_data = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_data)


def predict_word(seed_text, model, oov=False):
    text = [clean(seed_text)]
    sequence = tokenizer.texts_to_sequences(text)
    input_seq = np.array(pad_sequences(sequence, maxlen=model.input_shape[1], padding='pre'))
    if oov:
        y = model.predict(input_seq).argmax(axis=1)
        next_word = tokenizer.sequences_to_texts([y])
    else:
        y = model.predict(input_seq)[0, 2:].argmax(axis=0)
        next_word = tokenizer.sequences_to_texts([[y + 2]])
    return seed_text + ' | '+next_word[0]


def max_probs(input_seq, model, k=3, oov=False):
    pred = np.log(model.predict(input_seq))
    if oov:
        ids = np.argpartition(pred[0, :], -k)[-k:]
    else:
        ids = np.argpartition(pred[0, 2:], -k)[-k:]+2
    ids = np.expand_dims(ids, axis=1)
    log_probs = pred[0, :][ids]
    return ids, log_probs


def beam_search(seed_text, model, k=3, alpha=1, nwords=10, oov=False):
    df = pd.DataFrame(columns=['ids', 'log_probs'])
    maxlen = model.input_shape[1]
    # converting text to numeric vector
    text = [clean(seed_text)]
    sequence = tokenizer.texts_to_sequences(text)
    input0 = np.array(pad_sequences(sequence, maxlen=model.input_shape[1], padding='pre'))

    ids, log_probs = max_probs(input0, model, k=k, oov=oov)

    for i1 in range(k):
        df = df.append({'ids': ids[i1, :], 'log_probs': log_probs[i1, :]}, ignore_index=True)
    for nw in range(nwords):
        candidates = pd.DataFrame(columns=['ids', 'log_probs'])
        for i2 in range(k):
            lastk = df.iloc[-k:, :]
            word = lastk.iloc[i2, 0]
            new_input = np.hstack([input0[0], word])[word.shape[0]:].reshape(1, maxlen)
            old_log_probs = lastk.iloc[i2, 1]
            new_output = max_probs(new_input, model, k=k, oov=oov)
            for i3 in range(k):
                candidate_ids = np.hstack([new_input[0], new_output[0][i3, :]])[1:]
                candidate_log_probs = np.hstack([old_log_probs, new_output[1][i3, :]])
                candidates = candidates.append({'ids': candidate_ids[-nw - 2:], 'log_probs': candidate_log_probs},
                                               ignore_index=True)
        candidates['last_prob'] = candidates['log_probs'].apply(lambda x: x[-1])
        candidates = candidates.sort_values('last_prob').iloc[-k:, :2]
        df = df.append(candidates, ignore_index=True)

    df['Ty'] = df['log_probs'].apply(len)
    df['sum_log_prob'] = df['log_probs'].apply(sum)/df['Ty']**alpha
    winner = df.sort_values('sum_log_prob').iloc[-1, 0]

    if oov:
        predicted = tokenizer.sequences_to_texts([winner])[0]
    else:
        predicted = tokenizer.sequences_to_texts([winner])[0]

    predicted = predicted.replace('sentenceend', '.')
    return predicted
