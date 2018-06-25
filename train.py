import torch
import numpy as np
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim
from model import Model
from data_helpers import prepare_sequence
from data_helpers import load_training_data
from data_helpers import create_word2id
import random
import datetime
import os
import json
import time


if __name__ == '__main__':

    training_data = load_training_data('./data/train.json')
    question_word2id, relation_word2id = create_word2id(training_data)

    print(len(training_data))
    print(relation_word2id)
    print(question_word2id)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    dictionary_dir = os.path.abspath(os.path.join(out_dir, 'dictionary'))
    if not os.path.exists(dictionary_dir):
        os.makedirs(dictionary_dir)
    question_word2id_path = os.path.join(dictionary_dir, 'question_word2id.json')
    with open(question_word2id_path, 'w', encoding='utf-8') as fout:
        json.dump(question_word2id, fout, indent=2)
    relation_word2id_path = os.path.join(dictionary_dir, 'relation_word2id.json')
    with open(relation_word2id_path, 'w', encoding='utf-8') as fout:
        json.dump(relation_word2id, fout, indent=2)

    model = Model(
        relation_embedding_dim=100,
        relation_hidden_dim=200,
        relation_vocab_size=len(relation_word2id),
        question_embedding_dim=100,
        question_hidden_dim=200,
        question_vocab_size=len(question_word2id)
    )
    model.cuda()
    loss_function = nn.MarginRankingLoss(margin=0.5).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # for x in model.parameters():
    #     print(type(x.data), x.size())
    #     print(list(x.data))


    for epoch in range(1):
        accuracy = 0.0
        random.shuffle(training_data)
        step = 0
        for ques, pos, neg in training_data:
            model.zero_grad()
            question = prepare_sequence(ques.split(), question_word2id)

            positive_relation = prepare_sequence(pos.split(), relation_word2id)
            positive_word_level_relation = prepare_sequence(pos.replace('_', ' ').split(), relation_word2id)

            negative_relation = prepare_sequence(neg.split(), relation_word2id)
            negative_word_level_relation = prepare_sequence(neg.replace('_', ' ').split(), relation_word2id)

            pos_score, neg_score = model(question, positive_relation, positive_word_level_relation,
                                 negative_relation, negative_word_level_relation)
            loss = loss_function(pos_score.view(1), neg_score.view(1), torch.ones(1).cuda())
            step += 1
            print(ques, pos, neg)
            print(step, loss)
            if torch.gt(pos_score, neg_score):
                accuracy += 1
            loss.backward(retain_graph=True)
            optimizer.step()
            if step >= 2000:
                break
        time_str = datetime.datetime.now().isoformat()
        print('{}: step {}, acc {:g}'.format(time_str, epoch, accuracy / len(training_data)))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(model, os.path.join(checkpoint_dir, 'model.pth'))

