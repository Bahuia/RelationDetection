import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import Model
from data_helpers import prepare_sequence
from data_helpers import load_training_data
from data_helpers import create_word2id
from data_helpers import batch_iter
import numpy as np
import datetime
import os
import json
import time
import config


if __name__ == '__main__':

    # Load the training data.
    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = load_training_data(config.TRAIN_PATH)
    # Create the word2id dictionaries of questions and relations.
    dict_dir = os.path.abspath(os.path.join(os.path.curdir, 'dictionary'))
    que_word2id_path = os.path.abspath(os.path.join(dict_dir, 'question_word2id.json'))
    rel_word2id_path = os.path.abspath(os.path.join(dict_dir, 'relation_word2id.json'))
    with open(que_word2id_path, 'r', encoding='utf-8') as fin:
        que_word2id = json.load(fin)
    with open(rel_word2id_path, 'r', encoding='utf-8') as fin:
        rel_word2id = json.load(fin)

    print('Number of samples : {}'.format(len(qid)))
    print('Size of question word2id : {}'.format(len(que_word2id)))
    print('Size of relation word2id : {}'.format(len(rel_word2id)))

    # Change to pytorch Variable.
    que = prepare_sequence(que, 20, que_word2id)
    pos_rel = prepare_sequence(pos_rel, 2, rel_word2id)
    neg_rel = prepare_sequence(neg_rel, 2, rel_word2id)
    pos_rel_word = prepare_sequence(pos_rel_word, 15, rel_word2id)
    neg_rel_word = prepare_sequence(neg_rel_word, 15, rel_word2id)
    print(que.shape)
    print(pos_rel.shape)
    print(neg_rel.shape)
    print(pos_rel_word.shape)
    print(neg_rel_word.shape)

    # Create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create the model, loss function and optimizer.
    model = Model(
        relation_embedding_dim=config.RELATION_EMBEDDING_DIM,
        relation_hidden_dim=config.RELATION_HIDDEN_DIM,
        relation_vocab_size=len(rel_word2id),
        question_embedding_dim=config.QUESTION_EMBEDDING_DIM,
        question_hidden_dim=config.QUESTION_HIDDEN_DIM,
        question_vocab_size=len(que_word2id)
    )
    # model_path = os.path.abspath(os.path.join(os.path.curdir, 'runs', '1530371205'))
    # model = torch.load(os.path.join(model_path, 'checkpoints', 'model.pth'))
    # Use GPU.
    model.cuda()
    # Ust the loss function that the paper used.
    loss_function = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ones = Variable(torch.ones(1)).cuda()

    batches_train = batch_iter(
        data=list(zip(que, pos_rel, neg_rel, pos_rel_word, neg_rel_word)),
        batch_size=1024,
        num_epochs=config.EPOCH_NUM,
    )

    def train_step(batch):
        """
        A single training step
        """
        que_batch, pos_rel_batch, neg_rel_batch, pos_rel_word_batch, neg_rel_word_batch = zip(*batch)
        que_batch = torch.LongTensor(np.array(que_batch))
        pos_rel_batch = torch.LongTensor(np.array(pos_rel_batch))
        neg_rel_batch = torch.LongTensor(np.array(neg_rel_batch))
        pos_rel_word_batch = torch.LongTensor(np.array(pos_rel_word_batch))
        neg_rel_word_batch = torch.LongTensor(np.array(neg_rel_word_batch))
        # print(que_batch.size())
        # print(pos_rel_batch.size())
        # print(neg_rel_batch.size())
        # print(pos_rel_word_batch.size())
        # print(neg_rel_word_batch.size())
        model.zero_grad()
        accuracy = 0.0
        pos_score, neg_score = model(
            Variable(que_batch.cuda()),
            Variable(pos_rel_batch.cuda()),
            Variable(pos_rel_word_batch.cuda()),
            Variable(neg_rel_batch.cuda()),
            Variable(neg_rel_word_batch.cuda())
        )
        ones = torch.ones(len(que_batch)).cuda()
        loss = loss_function(pos_score, neg_score, ones)
        loss.backward()
        optimizer.step()
        for i in range(len(pos_score)):
            accuracy += 1.0 if torch.gt(pos_score[i], neg_score[i]) else 0.0
        accuracy /= len(batch)

        time_str = datetime.datetime.now().isoformat()
        print("{}: train step {}, loss {:g}, acc {:g}".format(time_str, current_step, loss, accuracy))

    # Training loop. For each batch...
    current_step = 0
    for batch in batches_train:
        train_step(batch)
        current_step += 1
        # Save model params
        if current_step % 100 == 0:
            path = os.path.join(checkpoint_dir, 'model.pth')
            torch.save(model, path)
            print("Saved model checkpoint to {}\n".format(path))

