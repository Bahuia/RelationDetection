import json
import torch
from torch.autograd import Variable
import numpy as np


def load_training_data(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = [], [], [], [], [], []
    for s in data:
        question = s['Pattern']
        pos_relation = s['ChainSamples']['PositiveSample']
        for neg_relation in s['ChainSamples']['NegativeSamples']:
            qid.append(int(s['QuestionId'][9 : len(s['QuestionId'])]))
            que.append(question)
            pos_rel.append(pos_relation)
            neg_rel.append(neg_relation)
            pos_rel_word.append(pos_relation.replace('_', ' '))
            neg_rel_word.append(neg_relation.replace('_', ' '))
    return qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word


def create_word2id(training_data):
    question_word2id = {'<unk>': 0}
    relation_word2id = {'<unk>': 0}
    for qid, question, positive_relation, negative_relation in training_data:
        for word in question.split():
            if word not in question_word2id:
                question_word2id[word] = len(question_word2id)

        for word in positive_relation.split():
            if word not in relation_word2id:
                relation_word2id[word] = len(relation_word2id)
        for word in positive_relation.replace('_', ' ').split():
            if word not in relation_word2id:
                relation_word2id[word] = len(relation_word2id)

        for word in negative_relation.split():
            if word not in relation_word2id:
                relation_word2id[word] = len(relation_word2id)
        for word in negative_relation.replace('_', ' ').split():
            if word not in relation_word2id:
                relation_word2id[word] = len(relation_word2id)
    return question_word2id, relation_word2id


def prepare_sequence(data, max_length,  word2id):
    idx_batch = []
    for seq in data:
        idxs = np.zeros(max_length)
        for i, x in enumerate(seq.split()):
            if x in word2id:
                idxs[i] = word2id[x]
        idx_batch.append(idxs)
    return np.array(idx_batch)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for a data set.
    :params shuffle: shuffle data or not
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1   # number of batches at each epoch
    for epoch in range(num_epochs):
        # shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]