import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import Model
from data_helpers import prepare_sequence
from data_helpers import load_data
from data_helpers import load_vocab
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
    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = load_data(config.TRAIN_PATH)
    qid_dev, que_dev, pos_rel_dev, pos_rel_word_dev, neg_rel_dev, neg_rel_word_dev = load_data(config.TEST_PATH)
    # Create the word2id dictionaries of questions and relations.
    que_word2id, rel_word2id = load_vocab(config.DICT_DIR)

    print('Size of question vocab : {}'.format(len(que_word2id)))
    print('Size of relation vocab : {}'.format(len(rel_word2id)))

    # Change to pytorch Variable.
    que = prepare_sequence(que, config.MAX_QUESTION_LENGTH, que_word2id)
    pos_rel = prepare_sequence(pos_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    neg_rel = prepare_sequence(neg_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    pos_rel_word = prepare_sequence(pos_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    neg_rel_word = prepare_sequence(neg_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    print('\nTrain set')
    print('question tensor shape: {}'.format(que.shape))
    print('positive relation level shape: {}'.format(pos_rel.shape))
    print('negative relation level shape: {}'.format(neg_rel.shape))
    print('positive word level shape: {}'.format(pos_rel_word.shape))
    print('negative word level shape: {}'.format(neg_rel_word.shape))

    que_dev = prepare_sequence(que_dev, config.MAX_QUESTION_LENGTH, que_word2id)
    pos_rel_dev = prepare_sequence(pos_rel_dev, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    neg_rel_dev = prepare_sequence(neg_rel_dev, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    pos_rel_word_dev = prepare_sequence(pos_rel_word_dev, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    neg_rel_word_dev = prepare_sequence(neg_rel_word_dev, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    print('\nDev set')
    print('question tensor shape: {}'.format(que_dev.shape))
    print('positive relation level shape: {}'.format(pos_rel_dev.shape))
    print('negative relation level shape: {}'.format(neg_rel_dev.shape))
    print('positive word level shape: {}'.format(pos_rel_word_dev.shape))
    print('negative word level shape: {}'.format(neg_rel_word_dev.shape))

    # Create runs directory.
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("\nWriting to {}\n".format(out_dir))

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
        question_vocab_size=len(que_word2id),
        using_gru=config.USING_GRU
    )
    # Shift model to GPU.
    model.cuda()
    # Ust the loss function that the paper used.
    loss_function = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ones = Variable(torch.ones(1)).cuda()

    batches_train = batch_iter(
        data=list(zip(que, pos_rel, neg_rel, pos_rel_word, neg_rel_word)),
        batch_size=config.TRAIN_BATCH_SIZE,
        num_epochs=config.EPOCH_NUM,
    )

    def train_step(batch):
        """
        A single train step for a batch.
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

    def dev_step(batch):
        """
        A single dev step for a batch.
        """
        qid, que_batch, pos_rel_batch, neg_rel_batch, pos_rel_word_batch, neg_rel_word_batch = zip(*batch)
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
        pos_score, neg_score = model(
            Variable(que_batch.cuda()),
            Variable(pos_rel_batch.cuda()),
            Variable(pos_rel_word_batch.cuda()),
            Variable(neg_rel_batch.cuda()),
            Variable(neg_rel_word_batch.cuda())
        )
        for i in range(len(batch)):
            # Set answering status of question False.
            if torch.ge(neg_score[i], pos_score[i]):
                result[qid[i][0]] = False


    best_accuracy = 0.0
    early_stop = False
    iters_not_improved = 0
    # Training loop. For each batch...
    current_step = 0
    model.train()
    for batch in batches_train:
        train_step(batch)
        current_step += 1
        # Save model params
        if current_step >= config.DEV_START_STEP and current_step % config.DEV_EVERY == 0:
            model.eval()
            # Set the answering status of each question True.
            result = {}
            for id in qid_dev:
                result[id] = True
            # Dev loop ...
            batches_dev = batch_iter(
                data=list(zip(np.array(qid_dev).reshape(len(qid_dev), -1),
                              que_dev, pos_rel_dev, neg_rel_dev, pos_rel_word_dev, neg_rel_word_dev)),
                batch_size=config.DEV_BATCH_SIZE,
                num_epochs=1,
            )
            for batch_dev in batches_dev:
                dev_step(batch_dev)
            # Calculate dev accuracy.
            accuracy = 0.0
            for key, value in result.items():
                accuracy += value
            accuracy /= config.QUESTION_NUMBER
            time_str = datetime.datetime.now().isoformat()
            print('\n{}: dev {} accuracy {:g}\n'.format(time_str, current_step, accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                path = os.path.join(checkpoint_dir, 'model.pth')
                torch.save(model, path)
                print("Saved model checkpoint to {}\n".format(path))
                iters_not_improved = 0
            else:
                iters_not_improved += 1
                if iters_not_improved > config.PATIENCE:
                    early_stop = True
                    break
            model.train()

    if early_stop:
        print('\nEarly stopped.')
    else:
        print('\nTraining finished.')