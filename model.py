import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    LSTM model for sentence to vector representation.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers, bidirectional):
        """
        Initialization of LSTM
        :param embedding_dim: dimension of the word embedding.
        :param hidden_dim: dimension of the hidden layer.
        :param vocab_size: vocabulary size.
        :param num_layers: number of the LSTM layers.
        :param bidirectional: is LSTM bidirectional.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

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
        hidden = (Variable(torch.zeros(self.num_layers * self.num_directions, batch_size , self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim)).cuda())
        embeds = self.word_embeddings(batch)
        # print('embeds', embeds.size())
        lstm_out, hidden = self.lstm(embeds, hidden)
        return lstm_out


class Model(nn.Module):
    """
    input: question, positive relation, negative relation.
    output: two cosine similarities.
    Model for compute the similarity of (question, positive relation) and (question, negative_relation).
    """

    def __init__(self, relation_embedding_dim, relation_hidden_dim, relation_vocab_size,
                 question_embedding_dim, question_hidden_dim, question_vocab_size, LAMBDA=0.5):
        """
        Initialization of the model.
        :param LAMBDA: the mini value that pos_sore bigger than neg_score.
        """
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
        """
        Calculate the cosine similarity of the question and the relation.
        :param question: question tensor.
        :param relation_level_relation: relation-level relation tensor.
        :param word_level_relation: word-level relation tensor.
        :return: cos(question, relation-level relation), cos(question, word-level relation).
        """
        # Question lstm layer.
        question_lstm_out = self.question_lstm(question)
        # Relation-level relation layer.
        relation_level_relation_lstm_out = self.relation_lstm(relation_level_relation)
        # Word-level relation layer.
        word_level_relation_lstm_out = self.relation_lstm(word_level_relation)
        #
        # print('question_lstm_out:', question_lstm_out.size())
        # print('relation_level_lstm_out:', relation_level_relation_lstm_out.size())
        # print('word_level_lstm_out:', word_level_relation_lstm_out.size())


        # Max-pooling question.
        question_max_pooling = nn.MaxPool2d(
            kernel_size=(question_lstm_out.size()[1], 1),
            stride=1
        )
        question_rep = question_max_pooling(question_lstm_out).view(len(question_lstm_out), -1)

        # Max-pooling relation.
        relation_lstm_out = torch.cat([relation_level_relation_lstm_out, word_level_relation_lstm_out], 1)
        relation_max_pooling = nn.MaxPool2d(
            kernel_size=(relation_lstm_out.size()[1], 1),
            stride=1
        )
        relation_rep = relation_max_pooling(relation_lstm_out).view(len(relation_lstm_out), -1)

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

    def forward(self, question, positive_relation, positive_word_level_relation, negative_relation, negative_word_level_relation):
        """
        Calculate two cosine similarities.
        """
        positive_score = self.similarity(question, positive_relation, positive_word_level_relation)
        negative_score = self.similarity(question, negative_relation, negative_word_level_relation)
        return positive_score, negative_score



