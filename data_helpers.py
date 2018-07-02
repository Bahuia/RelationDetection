import json
import torch
from torch.autograd import Variable


def load_training_data(path):
    with open(path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    training_data = []
    for s in data:
        question = s['Pattern']
        pos_relation = s['ChainSamples']['PositiveSample']
        for neg_relation in s['ChainSamples']['NegativeSamples']:
            sample = (s['QuestionId'], question, pos_relation, neg_relation)
            training_data.append(sample)
    return training_data


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


def prepare_sequence(sequence, word2id):
    idxs = []
    for x in sequence:
        if x not in word2id:
            idxs.append(word2id['<unk>'])
        else:
            idxs.append(word2id[x])
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)