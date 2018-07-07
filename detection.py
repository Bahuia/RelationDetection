import torch
from torch.autograd import Variable
from data_helpers import prepare_sequence
from data_helpers import load_data
from data_helpers import batch_iter
from data_helpers import load_vocab
from data_helpers import save_results
import config
import numpy as np
import datetime
import json
import os


if __name__ == '__main__':

    qid, _que, _pos_rel, _pos_rel_word, _neg_rel, _neg_rel_word = load_data(config.TEST_PATH)
    que_word2id, rel_word2id = load_vocab(config.DICT_DIR)

    id = []
    results = {}
    for i in range(len(qid)):
        id.append(i)
        results[qid[i]] = {
            'Question': _que[i],
            'Relation': _pos_rel[i],
            'PredictRelation': _pos_rel[i],
            'MaxPredictScore': -1.0,
            'Status': True
        }

    print('Number of test samples: {}'.format(len(qid)))
    print('Size of question word2id: {}'.format(len(que_word2id)))
    print('Size of relation word2id: {}'.format(len(rel_word2id)))

    # Change to pytorch Variable.
    que = prepare_sequence(_que, config.MAX_QUESTION_LENGTH, que_word2id)
    pos_rel = prepare_sequence(_pos_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    neg_rel = prepare_sequence(_neg_rel, config.MAX_RELATION_LEVEL_LENGTH, rel_word2id)
    pos_rel_word = prepare_sequence(_pos_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
    neg_rel_word = prepare_sequence(_neg_rel_word, config.MAX_WORD_LEVEL_LENGTH, rel_word2id)
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

    # Batch generator.
    batches_test = batch_iter(
        data=list(zip(np.array(id).reshape(len(id), -1), np.array(qid).reshape(len(qid), -1),
                      que, pos_rel, neg_rel, pos_rel_word, neg_rel_word)),
        batch_size=config.TEST_BATCH_SIZE,
        num_epochs=1,
        shuffle=False
    )

    # Test loop ...
    current_step = 0
    model.eval()
    for batch in batches_test:
        id, qid, que_batch, pos_rel_batch, neg_rel_batch, pos_rel_word_batch, neg_rel_word_batch = zip(*batch)
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
                results[qid[i][0]]['Status'] = False
                score = neg_score[i].cpu().detach().numpy()
                if score > results[qid[i][0]]['MaxPredictScore']:
                    results[qid[i][0]]['MaxPredictScore'] = score
                    results[qid[i][0]]['PredictRelation'] = _neg_rel[id[i][0]]
        current_step += 1
        print('Test step {}'.format(current_step))

    accuracy = 0.0
    for qid, result in results.items():
        accuracy += result['Status']
    accuracy /= config.QUESTION_NUMBER
    time_str = datetime.datetime.now().isoformat()
    print('\n{}: acc {:g}\n'.format(time_str, accuracy))


    results_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', config.DETECTION_MODEL))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_path = os.path.abspath(os.path.join(results_dir, 'result.csv'))
    save_results(results_path, results)
    print('Result save to {}\n'.format(results_path))