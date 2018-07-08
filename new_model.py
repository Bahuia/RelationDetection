import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable


class Rnn(nn.Module):
    """
    LSTM model for sentence to vector representation.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, bidirectional, using_gru=False):
        """
        Initialization of LSTM
        :param embedding_dim: dimension of the word embedding.
        :param hidden_dim: dimension of the hidden layer.
        :param vocab_size: vocabulary size.
        :param num_layers: number of the LSTM layers.
        :param bidirectional: is LSTM bidirectional.
        """
        super(Rnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.using_gru = using_gru

        # Word Embedding Layer
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        # RNN
        if self.using_gru == False:
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=True
            )

    def forward(self, batch):
        """
        Forward propagation of the model.
        :param word_sequence: input word sequence of the model
        :return: output of each time
        """

        batch_size = len(batch)
        embeds = self.word_embeddings(batch)
        # print('embeds', embeds.size())
        if self.using_gru == False:
            hidden = (Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)).cuda(),
                      Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)).cuda())
        else:
            hidden = Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)).cuda()
        rnn_out, hidden = self.rnn(embeds, hidden)
        return rnn_out


class Model(nn.Module):
    """
    input: question, positive relation, negative relation.
    output: two cosine similarities.
    Model for compute the similarity of (question, positive relation) and (question, negative_relation).
    """

    def __init__(self, relation_embedding_dim, relation_hidden_dim, relation_vocab_size,
                 question_embedding_dim, question_hidden_dim, question_vocab_size, num_layer = (1, 1),
                 bidirectional = True,  using_gru = False, LAMBDA=0.5):
        """
        Initialization of the model.
        :param LAMBDA: the mini value that pos_sore bigger than neg_score.
        """
        super(Model, self).__init__()
        self.LAMBDA = LAMBDA
        relation_num_layers, question_num_layers = num_layer

        # Relation RNN
        self.relation_encoder = Rnn(
            embedding_dim=relation_embedding_dim,
            hidden_dim=relation_hidden_dim,
            vocab_size=relation_vocab_size,
            num_layers=relation_num_layers,
            bidirectional=bidirectional,
            using_gru = using_gru
        )
        # Question RNN
        self.question_encoder = Rnn(
            embedding_dim=question_embedding_dim,
            hidden_dim=question_hidden_dim,
            vocab_size=question_vocab_size,
            num_layers=question_num_layers,
            bidirectional=bidirectional,
            using_gru = using_gru
        )

    def similarity(self, question, name_lv_rel, word_lv_rel, char_lv_question=None, char_lv_rel=None):
        """
        Calculate the cosine similarity of the question and the relation.
        :param question: question tensor.
        :param relation_level_relation: relation-level relation tensor.
        :param word_level_relation: word-level relation tensor.
        :return: cos(question, relation-level relation), cos(question, word-level relation).
        """
        # Question lstm layer.
        question_rnn_out = self.question_encoder(question)
        char_lv_question_rnn_out = self.question_encoder(char_lv_question)
        # Relation-level relation layer.
        name_lv_rel_rnn_out = self.relation_encoder(name_lv_rel)
        # Word-level relation layer.
        word_lv_rel_rnn_out = self.relation_encoder(word_lv_rel)
        char_lv_rel_rnn_out = self.relation_encoder(char_lv_rel)
        #
        # print('question_rnn_out:', question_rnn_out.size())
        # print('relation_level_lstm_out:', relation_level_relation_lstm_out.size())
        # print('word_level_lstm_out:', word_level_relation_lstm_out.size())


        # Max-pooling question.
        # if char_lv_question != None:
        question_rnn_out = torch.cat([question_rnn_out, char_lv_question_rnn_out], 1)
        question_max_pooling = nn.MaxPool2d(
            kernel_size=(question_rnn_out.size()[1], 1),
            stride=1
        )
        question_rep = question_max_pooling(question_rnn_out).view(len(question_rnn_out), -1)

        # Max-pooling relation.
        relation_rnn_out = torch.cat([name_lv_rel_rnn_out, word_lv_rel_rnn_out], 1)
        # Add character-level representation.
        # if char_lv_rel != None:
        relation_rnn_out = torch.cat([relation_rnn_out, char_lv_rel_rnn_out], 1)

        relation_max_pooling = nn.MaxPool2d(
            kernel_size=(relation_rnn_out.size()[1], 1),
            stride=1
        )
        relation_rep = relation_max_pooling(relation_rnn_out).view(len(relation_rnn_out), -1)

        # print('question_rep:', question_rep.size())
        # print('relation_rep:', relation_rep.size())

        # Cosine Similarity.
        xx = torch.sum(torch.mul(question_rep, relation_rep), 1)
        yy = torch.mul(torch.sqrt(torch.sum(torch.mul(question_rep, question_rep), 1)),
                                torch.sqrt(torch.sum(torch.mul(relation_rep, relation_rep), 1)))
        cosine = torch.div(xx, yy)
        # print('xx:', xx.size())
        # print('yy:', yy.size())
        # print('cos:', cosine.size())
        return cosine

    def forward(self, question, pos_rel, neg_rel):
        """
        Calculate two cosine similarities.
        """
        word_lv_question, char_lv_question = question
        pos_name_lv_rel, pos_word_lv_rel, pos_char_lv_rel = pos_rel
        pos_score = self.similarity(
            word_lv_question, pos_name_lv_rel, pos_word_lv_rel, char_lv_question, pos_char_lv_rel)
        neg_name_lv_rel, neg_word_lv_rel, neg_char_lv_rel = neg_rel
        neg_score = self.similarity(
            word_lv_question, neg_name_lv_rel, neg_word_lv_rel, char_lv_question, neg_char_lv_rel)
        return pos_score, neg_score



