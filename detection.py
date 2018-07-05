import torch
from torch.autograd import Variable
from data_helpers import prepare_sequence
from data_helpers import load_data
from data_helpers import batch_iter
from data_helpers import load_vocab
import config
import numpy as np
import datetime
import json
import os


if __name__ == '__main__':

    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = load_data(config.TEST_PATH)
    que_word2id, rel_word2id = load_vocab(config.DICT_DIR)

    print('Number of test samples: {}'.format(len(qid)))
    print('Size of question word2id: {}'.format(len(que_word2id)))
    print('Size of relation word2id: {}'.format(len(rel_word2id)))

    # Change to pytorch Variable.
    que = prepare_sequence(que, config.MAX_QUESTION_LENGTH, que_word2id)
    pos_rel = prepare_sequence(pos_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    neg_rel = prepare_sequence(neg_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    pos_rel_word = prepare_sequence(pos_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    neg_rel_word = prepare_sequence(neg_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    print('question tensor shape: {}'.format(que.shape))
    print('positive relation level shape: {}'.format(pos_rel.shape))
    print('negative relation level shape: {}'.format(neg_rel.shape))
    print('positive word level shape: {}'.format(pos_rel_word.shape))
    print('negative word level shape: {}'.format(neg_rel_word.shape))

    # Load model
    model = torch.load(config.MODEL_PATH)
    model.cuda()

    print('\nParameters of model')
    for name, x in model.named_parameters():
        print(name, x.size())
    print()

    # Set the answering status of each question True.
    result = {}
    for id in qid:
        result[id] = True
    print('Number of questions: {}'.format(len(result)))

    # Batch generator.
    batches_test = batch_iter(
        data=list(zip(np.array(qid).reshape(len(qid), -1), que, pos_rel, neg_rel, pos_rel_word, neg_rel_word)),
        batch_size=config.TEST_BATCH_SIZE,
        num_epochs=1,
    )

    # Test loop ...
    current_step = 0
    model.eval()
    for batch in batches_test:
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
        current_step += 1
        print('Test step {}'.format(current_step))

    accuracy = 0.0
    print('\nQuestion answering status:')
    for key, value in result.items():
        print('WebQTest-{} : {}'. format(key, value))
        accuracy += value
    accuracy /= config.QUESTION_NUMBER

    time_str = datetime.datetime.now().isoformat()
    print('{}: acc {:g}'.format(time_str, accuracy))