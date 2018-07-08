import os
import json
import numpy as np
import csv
import config
import torch



def create_vocab(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    question_vocab = {'<unk>': 0}
    relation_vocab = {'<unk>': 0}
    for s in data:
        question = s['Pattern']
        for word in question.split(' '):
            if word not in question_vocab:
                question_vocab[word] = len(question_vocab)
        for char in question:
            if char not in question_vocab:
                question_vocab[char] = len(question_vocab)

        pos_rel = s['ChainSamples']['PositiveSample']
        for name in pos_rel.split(' '):
            if name not in relation_vocab:
                relation_vocab[name] = len(relation_vocab)
        for word in pos_rel.replace('_', ' ').split(' '):
            if word not in relation_vocab:
                relation_vocab[word] = len(relation_vocab)
        for char in pos_rel:
            if char not in relation_vocab:
                relation_vocab[char] = len(relation_vocab)

        for neg_rel in s['ChainSamples']['NegativeSamples']:
            for name in neg_rel.split(' '):
                if name not in relation_vocab:
                    relation_vocab[name] = len(relation_vocab)
            for word in neg_rel.replace('_', ' ').split(' '):
                if word not in relation_vocab:
                    relation_vocab[word] = len(relation_vocab)
            for char in neg_rel:
                if char not in relation_vocab:
                    relation_vocab[char] = len(relation_vocab)
    torch.save(question_vocab, './dictionary/question_vocab')
    torch.save(relation_vocab, './dictionary/relation_vocab')


def load_data(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = [], [], [], [], [], []
    for s in data:
        question = s['Pattern']
        pos_relation = s['ChainSamples']['PositiveSample']
        for neg_relation in s['ChainSamples']['NegativeSamples']:
            pos = s['QuestionId'].find('-')
            qid.append(int(s['QuestionId'][pos + 1 : len(s['QuestionId'])]))
            que.append(question)
            pos_rel.append(pos_relation)
            neg_rel.append(neg_relation)
            pos_rel_word.append(pos_relation.replace('_', ' '))
            neg_rel_word.append(neg_relation.replace('_', ' '))
    return qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word


def _load_data(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    qid, que, que_char, pos_rel_name, pos_rel_word, pos_rel_char, \
    neg_rel_name, neg_rel_word, neg_rel_char = [], [], [], [], [], [], [], [], []
    for s in data:
        question = s['Pattern']
        pos_relation = s['ChainSamples']['PositiveSample']
        for neg_relation in s['ChainSamples']['NegativeSamples']:
            pos = s['QuestionId'].find('-')
            qid.append(int(s['QuestionId'][pos + 1 : len(s['QuestionId'])]))
            que.append(question)
            que_string = ''
            for i, c in enumerate(question):
                if i: que_string += ' '
                que_string += c
            que_char.append(que_string)
            pos_rel_name.append(pos_relation)
            neg_rel_name.append(neg_relation)
            pos_rel_word.append(pos_relation.replace('_', ' '))
            neg_rel_word.append(neg_relation.replace('_', ' '))
            pos_char = ''
            for i, c in enumerate(pos_relation):
                if i: pos_char += ' '
                pos_char += c
            pos_rel_char.append(pos_char)
            neg_char = ''
            for i, c in enumerate(neg_relation):
                if i: neg_char += ' '
                neg_char += c
            neg_rel_char.append(neg_char)
            # print(pos_char)
            # print(neg_char)
    return qid, que, que_char, pos_rel_name, pos_rel_word, pos_rel_char, neg_rel_name, neg_rel_word, neg_rel_char


def load_vocab(dict_dir):
    que_word2id_path = os.path.abspath(os.path.join(dict_dir, 'question_word2id.json'))
    rel_word2id_path = os.path.abspath(os.path.join(dict_dir, 'relation_word2id.json'))
    with open(que_word2id_path, 'r', encoding='utf-8') as fin:
        que_word2id = json.load(fin)
    with open(rel_word2id_path, 'r', encoding='utf-8') as fin:
        rel_word2id = json.load(fin)
    return que_word2id, rel_word2id


def _load_vocab(dict_dir):
    question_vocab_path = os.path.abspath(os.path.join(dict_dir, 'question_vocab'))
    relation_vocab_path = os.path.abspath(os.path.join(dict_dir, 'relation_vocab'))
    question_vocab = torch.load(question_vocab_path)
    relation_vocab = torch.load(relation_vocab_path)
    return question_vocab, relation_vocab


def prepare_sequence(data, max_length,  word2id):
    """
    Change word sequence to the id sequence.
    """
    idx_batch = []
    for seq in data:
        # if len(seq.split()) > max_length:
        #     print(seq)
        idxs = np.zeros(max_length)
        for i, x in enumerate(seq.split()):
            if x in word2id:
                idxs[i] = word2id[x]
        idx_batch.append(idxs)
    return np.array(idx_batch)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generate a batch iterator for a data set.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1   # number of batches at each epoch
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


def save_results(path, results):
    with open(path, 'w', encoding='utf-8') as fout:
        writer = csv.writer(fout, dialect='excel')
        head = ['QuestionId', 'Question', 'Relation', 'PredictRelation', 'Status']
        writer.writerow(head)
        for qid, result in results.items():
            row = ['WebQTest-' + str(qid), result[head[1]], result[head[2]], result[head[3]], result[head[4]]]
            writer.writerow(row)


if __name__ == '__main__':
    create_vocab(config.TRAIN_PATH)
    question_vocab = torch.load('./dictionary/question_vocab')
    print(question_vocab)