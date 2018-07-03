import torch
from torch.autograd import Variable
from data_helpers import prepare_sequence
from data_helpers import load_training_data
from data_helpers import batch_iter
import numpy as np
import datetime
import json
import os


if __name__ == '__main__':

    qid, que, pos_rel, pos_rel_word, neg_rel, neg_rel_word = load_training_data('./data/test.json')
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

    model_path = os.path.abspath(os.path.join(os.path.curdir, 'runs', '1530541749'))
    model = torch.load(os.path.join(model_path, 'checkpoints', 'model.pth'))
    model.cuda()

    for name, x in model.named_parameters():
        print(name, x.size())

    result = {}
    for id in qid:
        result[id] = 1
    print(len(result))

    batches_train = batch_iter(
        data=list(zip(np.array(qid).reshape(len(qid), -1), que, pos_rel, neg_rel, pos_rel_word, neg_rel_word)),
        batch_size=1024,
        num_epochs=1,
    )

    cnt = 0
    for batch in batches_train:
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
        model.zero_grad()
        accuracy = 0.0
        pos_score, neg_score = model(
            Variable(que_batch.cuda()),
            Variable(pos_rel_batch.cuda()),
            Variable(pos_rel_word_batch.cuda()),
            Variable(neg_rel_batch.cuda()),
            Variable(neg_rel_word_batch.cuda())
        )
        for i in range(len(batch)):
            if torch.ge(neg_score[i], pos_score[i]):
                result[qid[i][0]] = 0
        cnt += 1
        print(cnt)

    accuracy = 0.0
    for key, value in result.items():
        print(key, value)
        accuracy += value

    time_str = datetime.datetime.now().isoformat()
    print('{}: acc {:g}'.format(time_str, accuracy / 1639))