import torch
from data_helpers import prepare_sequence
from data_helpers import load_training_data
import datetime
import json
import os


if __name__ == '__main__':

    test_data = load_training_data('./data/test.json')
    model_path = os.path.abspath(os.path.join(os.path.curdir, 'runs', '1530106235'))
    question_word2id_path = os.path.abspath(os.path.join(model_path, 'dictionary', 'question_word2id.json'))
    relation_word2id_path = os.path.abspath(os.path.join(model_path, 'dictionary', 'relation_word2id.json'))
    with open(question_word2id_path, 'r', encoding='utf-8') as fin:
        question_word2id = json.load(fin)
    with open(relation_word2id_path, 'r', encoding='utf-8') as fin:
        relation_word2id = json.load(fin)

    print(len(test_data))
    print(relation_word2id)
    print(question_word2id)

    model = torch.load(os.path.join(model_path, 'checkpoints', 'model.pth'))
    model.cuda()

    # for x in model.parameters():
    #     print(type(x.data), x.size())
    #     print(list(x.data))

    result = {}
    for qid, ques, pos, neg in test_data:
        result[qid] = 1
    print(len(result))

    step = 0
    for qid, ques, pos, neg in test_data:
        question = prepare_sequence(ques.split(), question_word2id)

        positive_relation = prepare_sequence(pos.split(), relation_word2id)
        positive_word_level_relation = prepare_sequence(pos.replace('_', ' ').split(), relation_word2id)

        negative_relation = prepare_sequence(neg.split(), relation_word2id)
        negative_word_level_relation = prepare_sequence(neg.replace('_', ' ').split(), relation_word2id)

        pos_score, neg_score = model(question, positive_relation, positive_word_level_relation,
                             negative_relation, negative_word_level_relation)
        step += 1
        # print('step {}: ques: "{}" | pos: "{}" | wpos: "{}" | neg: "{}" | wneg: "{}"'.format(
        #     step, ques, pos, pos.replace('_', ' '), neg, neg.replace('_', ' ')))
        if torch.ge(neg_score, pos_score):
            result[qid] = 0

    accuracy = 0.0
    for key, value in result.items():
        print(key, value)
        accuracy += value

    time_str = datetime.datetime.now().isoformat()
    print('{}: acc {:g}'.format(time_str, accuracy / len(result)))