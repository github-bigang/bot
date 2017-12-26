import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
import os

# Assuming the label ID is starting from 1
def readFile(f):
    d = {}
    maxID = -1
    with open(f, 'r') as data:
        for line in data:
            fields = line.strip().split('\t')
            i = int(fields[0])
            d[fields[1]] = i
            if i > maxID:
                maxID = i
    return d, maxID

# Assuming the label ID is starting from 1
def load_data_and_labels(word_vocab, intentSegFile, vocab_processor=None, out_dir=None):
    all_intents, maxLabelID = readFile(intentSegFile)
    texts, labels = [], []
    label_dist = [0] * maxLabelID
    all_chars = []
    for intent in all_intents:
        word_idx = word_vocab.to_index_sequence(' '.join([x.decode('utf-8') for x in intent.split(' ')]))
        texts.append(word_idx)
        label = [0] * (maxLabelID)
        l = all_intents[intent] - 1
        label[l] = 1
        labels.append(label)
        label_dist[l] += 1
        chars = ''.join([x + ' ' for x in intent.replace(' ', '').decode('utf-8')])
        all_chars.append(chars)
    y = np.concatenate([labels], 0)
    print('label distribution: ' + ' '.join([str(x) for x in label_dist]))
    # Build vocabulary
    max_document_length = max([len(x) for x in texts])
    if max_document_length > 50: max_document_length = 50
    max_document_char_length = max([len(x.split(" ")) for x in all_chars])
    if max_document_char_length > 100: max_document_char_length = 100
    if not vocab_processor:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_char_length)
        vocab_processor.save(os.path.join(out_dir, "vocab"))
    x_chars = np.array(list(vocab_processor.fit_transform(all_chars)))
    print('max sentence length: ' + str(max_document_length) + ' max sentence char length: ' + str(max_document_char_length))
    return texts, y, x_chars, max_document_length, max_document_char_length, len(vocab_processor.vocabulary_)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
