import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim


class LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, bidirectional):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden = self.init_hidden()

        # Word Embedding Layer
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        # Unit of LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

    def init_hidden(self):
        """
        Initialization of the LSTM hidden layer.
        :return: initialized hidden_vector and output_vector
        """
        return (autograd.Variable(torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_dim)).cuda())

    def forward(self, word_sequence):
        """
        Forward propagation of the model.
        :param word_sequence: input word sequence of the model
        :return: output of each time
        """
        embeds = self.word_embeddings(word_sequence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(embeds), 1, -1), self.hidden)
        return lstm_out


class Model(nn.Module):

    def __init__(self, relation_embedding_dim, relation_hidden_dim, relation_vocab_size,
                 question_embedding_dim, question_hidden_dim, question_vocab_size, LAMBDA=0.5):
        super(Model, self).__init__()
        self.LAMBDA = LAMBDA

        # Relation LSTM
        self.relation_lstm = LSTM(
            embedding_dim=relation_embedding_dim,
            hidden_dim=relation_hidden_dim,
            vocab_size=relation_vocab_size,
            num_layers=1,
            bidirectional=True
        )
        # Question LSTM
        self.question_lstm = LSTM(
            embedding_dim=question_embedding_dim,
            hidden_dim=question_hidden_dim,
            vocab_size=question_vocab_size,
            num_layers=1,
            bidirectional=True
        )

    def similarity(self, question, relation_level_relation, word_level_relation):
        self.question_lstm.hidden = self.question_lstm.init_hidden()
        question_lstm_out = self.question_lstm(question)

        self.relation_lstm.hidden = self.relation_lstm.init_hidden()
        relation_level_relation_lstm_out = self.relation_lstm(relation_level_relation).view(len(relation_level_relation), -1)

        self.relation_lstm.hidden = self.relation_lstm.init_hidden()
        word_level_relation_lstm_out = self.relation_lstm(word_level_relation).view(len(word_level_relation), -1)

        # Max-pooling question
        question_max_pooling = nn.MaxPool2d(
            kernel_size=(len(question_lstm_out), 1),
            stride=1
        )
        question_rep = question_max_pooling(question_lstm_out.view(1, len(question_lstm_out), -1)).view(-1)

        # Max-pooling relation
        relation_lstm_out = torch.cat(
            [relation_level_relation_lstm_out, word_level_relation_lstm_out], 0)
        relation_max_pooling = nn.MaxPool2d(
            kernel_size=(len(relation_lstm_out), 1),
            stride=1
        )
        relation_rep = relation_max_pooling(relation_lstm_out.view(1, len(relation_lstm_out), -1)).view(-1)

        # print('question rep', question_rep)
        # print('relation rep', relation_rep)

        # Cosine Similarity
        numerator = torch.sum(question_rep * relation_rep)
        denominator = torch.sqrt(torch.sum(question_rep * question_rep)) *  \
                      torch.sqrt(torch.sum(relation_rep * relation_rep))
        cosine = numerator / denominator
        return cosine

    def forward(self, question, positive_relation, positive_word_level_relation, negative_relation, negative_word_level_relation):
        positive_score = self.similarity(question, positive_relation, positive_word_level_relation)
        negative_score = self.similarity(question, negative_relation, positive_word_level_relation)
        return positive_score, negative_score


