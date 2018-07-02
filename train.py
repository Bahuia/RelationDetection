import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import Model
from data_helpers import prepare_sequence
from data_helpers import load_training_data
from data_helpers import create_word2id
import random
import datetime
import os
import json
import time
import config


if __name__ == '__main__':
    # Load the training data.
    training_data = load_training_data(config.TRAIN_PATH)
    random.shuffle(training_data)
    # Create the word2id dictionaries of questions and relations.
    question_word2id, relation_word2id = create_word2id(training_data)

    print('Number of samples : {}'.format(len(training_data)))
    print('Size of question word2id : {}'.format(len(question_word2id)))
    print('Size of relation word2id : {}'.format(len(relation_word2id)))

    # Create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Save two word2ids.
    dictionary_dir = os.path.abspath(os.path.join(out_dir, 'dictionary'))
    if not os.path.exists(dictionary_dir):
        os.makedirs(dictionary_dir)
    question_word2id_path = os.path.join(dictionary_dir, 'question_word2id.json')
    json.dump(question_word2id, open(question_word2id_path, 'w', encoding='utf-8'), indent=2)
    relation_word2id_path = os.path.join(dictionary_dir, 'relation_word2id.json')
    json.dump(relation_word2id, open(relation_word2id_path, 'w', encoding='utf-8'), indent=2)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    qid, ques, pos, pos_w, neg, neg_w = [], [], [], [], [], []
    for id, q, p, n in training_data:
        qid.append(id)
        # prepare question tensor.
        ques.append(prepare_sequence(q.split(), question_word2id))
        # prepare positive relation tensor.
        pos.append(prepare_sequence(p.split(), relation_word2id))
        pos_w.append(prepare_sequence(p.replace('_', ' ').split(), relation_word2id))
        # prepare negative relation tensor.
        neg.append(prepare_sequence(n.split(), relation_word2id))
        neg_w.append(prepare_sequence(n.replace('_', ' ').split(), relation_word2id))
    del training_data

    # Create the model, loss function and optimizer.
    model = Model(
        relation_embedding_dim=config.RELATION_EMBEDDING_DIM,
        relation_hidden_dim=config.RELATION_HIDDEN_DIM,
        relation_vocab_size=len(relation_word2id),
        question_embedding_dim=config.QUESTION_EMBEDDING_DIM,
        question_hidden_dim=config.QUESTION_HIDDEN_DIM,
        question_vocab_size=len(question_word2id)
    )
    # Use GPU.
    model.cuda()
    # Ust the loss function that the paper used.
    loss_function = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ones = Variable(torch.ones(1)).cuda()

    # Start training ...
    time_str = datetime.datetime.now().isoformat()
    print('Start training at ' + time_str)
    for epoch in range(config.EPOCH_NUM):
        accuracy = 0.0
        step = 0
        last_time = datetime.datetime.now()
        for i in range(len(qid)):
            # clean all gradients.
            model.zero_grad()

            # calculate the positive similarity score and negative similarity score.
            pos_score, neg_score = model(ques[i].cuda(), pos[i].cuda(), pos_w[i].cuda(), neg[i].cuda(), neg_w[i].cuda())
            loss = loss_function(pos_score, neg_score, ones)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 1000 == 0:
                now_time = datetime.datetime.now()
                print(step, (now_time - last_time).seconds)
                last_time = now_time

            accuracy += 1 if torch.gt(pos_score, neg_score) else 0

        time_str = datetime.datetime.now().isoformat()
        print('{}: epoch {}, acc {:g}'.format(time_str, epoch, accuracy / len(qid)))
        torch.save(model, os.path.join(checkpoint_dir, 'model.pth'))

