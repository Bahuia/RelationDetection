import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from new_model import Model
from data_helpers import prepare_sequence
from data_helpers import _load_data as load_data
from data_helpers import _load_vocab as load_vocab
from data_helpers import batch_iter
import numpy as np
import datetime
import os
import json
import time
import config


if __name__ == '__main__':

    # Load the training data.
    qid, que_word, que_char, pos_rel_name, pos_rel_word, pos_rel_char, \
    neg_rel_name, neg_rel_word, neg_rel_char = load_data(config.TRAIN_PATH)
    qid_dev, que_word_dev, que_char_dev, pos_rel_name_dev, pos_rel_word_dev, pos_rel_char_dev, \
    neg_rel_name_dev, neg_rel_word_dev, neg_rel_char_dev = load_data(config.TEST_PATH)
    # Create the word2id dictionaries of questions and relations.
    que_vocab, rel_vocab = load_vocab(config.DICT_DIR)

    print('Size of question vocab : {}'.format(len(que_vocab)))
    print('Size of relation vocab : {}'.format(len(rel_vocab)))

    # Change to pytorch Variable.
    que_word = prepare_sequence(que_word, config.MAX_QUESTION_LENGTH, que_vocab)
    que_char = prepare_sequence(que_char, config.MAX_QUESTION_CHAR_LEVEL_LENGTH, que_vocab)
    pos_rel_name = prepare_sequence(pos_rel_name, config.MAX_RELATION_LEVEL_LENGTH, rel_vocab)
    neg_rel_name = prepare_sequence(neg_rel_name, config.MAX_RELATION_LEVEL_LENGTH, rel_vocab)
    pos_rel_word = prepare_sequence(pos_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_vocab)
    neg_rel_word = prepare_sequence(neg_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_vocab)
    pos_rel_char = prepare_sequence(pos_rel_char, config.MAX_CHAR_LEVEL_LENGTH, rel_vocab)
    neg_rel_char = prepare_sequence(neg_rel_char, config.MAX_CHAR_LEVEL_LENGTH, rel_vocab)
    print('\nTrain set')
    print('question word-level tensor shape: {}'.format(que_word.shape))
    print('question char-level tensor shape: {}'.format(que_char.shape))
    print('positive relation name-level tensor shape: {}'.format(pos_rel_name.shape))
    print('negative relation name-level tensor shape: {}'.format(neg_rel_name.shape))
    print('positive relation word-level tensor shape: {}'.format(pos_rel_word.shape))
    print('negative relation word-level tensor shape: {}'.format(neg_rel_word.shape))
    print('positive relation char-level tensor shape: {}'.format(pos_rel_char.shape))
    print('negative relation char-level tensor shape: {}'.format(neg_rel_char.shape))

    que_word_dev = prepare_sequence(que_word_dev, config.MAX_QUESTION_LENGTH, que_vocab)
    que_char_dev = prepare_sequence(que_char_dev, config.MAX_QUESTION_CHAR_LEVEL_LENGTH, que_vocab)
    pos_rel_name_dev = prepare_sequence(pos_rel_name_dev, config.MAX_RELATION_LEVEL_LENGTH, rel_vocab)
    neg_rel_name_dev = prepare_sequence(neg_rel_name_dev, config.MAX_RELATION_LEVEL_LENGTH, rel_vocab)
    pos_rel_word_dev = prepare_sequence(pos_rel_word_dev, config.MAX_WORD_LEVEL_LENGTH, rel_vocab)
    neg_rel_word_dev = prepare_sequence(neg_rel_word_dev, config.MAX_WORD_LEVEL_LENGTH, rel_vocab)
    pos_rel_char_dev = prepare_sequence(pos_rel_char_dev, config.MAX_CHAR_LEVEL_LENGTH, rel_vocab)
    neg_rel_char_dev = prepare_sequence(neg_rel_char_dev, config.MAX_CHAR_LEVEL_LENGTH, rel_vocab)
    print('\nDev set')
    print('question word-level tensor shape: {}'.format(que_word_dev.shape))
    print('question char-level tensor shape: {}'.format(que_char_dev.shape))
    print('positive relation name-level tensor shape: {}'.format(pos_rel_name_dev.shape))
    print('negative relation name-level tensor shape: {}'.format(neg_rel_name_dev.shape))
    print('positive relation word-level tensor shape: {}'.format(pos_rel_word_dev.shape))
    print('negative relation word-level tensor shape: {}'.format(neg_rel_word_dev.shape))
    print('positive relation char-level tensor shape: {}'.format(pos_rel_char_dev.shape))
    print('negative relation char-level tensor shape: {}'.format(neg_rel_char_dev.shape))

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
        relation_vocab_size=len(rel_vocab),
        question_embedding_dim=config.QUESTION_EMBEDDING_DIM,
        question_hidden_dim=config.QUESTION_HIDDEN_DIM,
        question_vocab_size=len(que_vocab),
        using_gru=config.USING_GRU
    )
    # Shift model to GPU.
    model.cuda()
    # Ust the loss function that the paper used.
    loss_function = nn.MarginRankingLoss(margin=config.MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    ones = Variable(torch.ones(1)).cuda()

    batches_train = batch_iter(
        data=list(zip(que_word, que_char, pos_rel_name, neg_rel_name,
                      pos_rel_word, neg_rel_word, pos_rel_char, neg_rel_char)),
        batch_size=config.TRAIN_BATCH_SIZE,
        num_epochs=config.EPOCH_NUM,
    )

    def train_step(batch):
        """
        A single train step for a batch.
        """
        que_word_batch, que_char_batch, pos_rel_name_batch, neg_rel_name_batch, pos_rel_word_batch, \
        neg_rel_word_batch, pos_rel_char_batch, neg_rel_char_batch = zip(*batch)
        que_word_batch = torch.LongTensor(np.array(que_word_batch))
        que_char_batch = torch.LongTensor(np.array(que_char_batch))
        pos_rel_name_batch = torch.LongTensor(np.array(pos_rel_name_batch))
        neg_rel_name_batch = torch.LongTensor(np.array(neg_rel_name_batch))
        pos_rel_word_batch = torch.LongTensor(np.array(pos_rel_word_batch))
        neg_rel_word_batch = torch.LongTensor(np.array(neg_rel_word_batch))
        pos_rel_char_batch = torch.LongTensor(np.array(pos_rel_char_batch))
        neg_rel_char_batch = torch.LongTensor(np.array(neg_rel_char_batch))
        # print(que_batch.size())
        # print(pos_rel_batch.size())
        # print(neg_rel_batch.size())
        # print(pos_rel_word_batch.size())
        # print(neg_rel_word_batch.size())
        model.zero_grad()
        accuracy = 0.0
        pos_score, neg_score = model(
            [Variable(que_word_batch.cuda()), Variable(que_char_batch.cuda())],
            [Variable(pos_rel_name_batch.cuda()),
             Variable(pos_rel_word_batch.cuda()),
             Variable(pos_rel_char_batch.cuda())],
            [Variable(neg_rel_name_batch.cuda()),
             Variable(neg_rel_word_batch.cuda()),
             Variable(neg_rel_char_batch.cuda())]
        )
        ones = torch.ones(len(que_word_batch)).cuda()
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
        qid, que_word_batch, que_char_batch, pos_rel_name_batch, neg_rel_name_batch, pos_rel_word_batch, \
        neg_rel_word_batch, pos_rel_char_batch, neg_rel_char_batch = zip(*batch)
        que_word_batch = torch.LongTensor(np.array(que_word_batch))
        que_char_batch = torch.LongTensor(np.array(que_char_batch))
        pos_rel_name_batch = torch.LongTensor(np.array(pos_rel_name_batch))
        neg_rel_name_batch = torch.LongTensor(np.array(neg_rel_name_batch))
        pos_rel_word_batch = torch.LongTensor(np.array(pos_rel_word_batch))
        neg_rel_word_batch = torch.LongTensor(np.array(neg_rel_word_batch))
        pos_rel_char_batch = torch.LongTensor(np.array(pos_rel_char_batch))
        neg_rel_char_batch = torch.LongTensor(np.array(neg_rel_char_batch))
        # print(que_batch.size())
        # print(pos_rel_batch.size())
        # print(neg_rel_batch.size())
        # print(pos_rel_word_batch.size())
        # print(neg_rel_word_batch.size())
        pos_score, neg_score = model(
            [Variable(que_word_batch.cuda()), Variable(que_char_batch.cuda())],
            [Variable(pos_rel_name_batch.cuda()),
             Variable(pos_rel_word_batch.cuda()),
             Variable(pos_rel_char_batch.cuda())],
            [Variable(neg_rel_name_batch.cuda()),
            Variable(neg_rel_word_batch.cuda()),
            Variable(neg_rel_char_batch.cuda())]
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
                              que_word_dev, que_char_dev, pos_rel_name_dev,
                              neg_rel_name_dev, pos_rel_word_dev, neg_rel_word_dev,
                              pos_rel_char_dev, neg_rel_char_dev)),
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