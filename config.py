# Config for project
import os

# Model label
DETECTION_MODEL = 'gru_l2_smart'
# Path of the training data.
TRAIN_PATH = os.path.abspath(os.path.join(os.path.curdir, 'data', 'train.data'))
# Path of the test data.
TEST_PATH = os.path.abspath(os.path.join(os.path.curdir, 'data', 'test.smart.data'))
DICT_DIR = os.path.abspath(os.path.join(os.path.curdir, 'dictionary'))
MODEL_PATH = os.path.abspath(os.path.join(os.path.curdir, 'runs', DETECTION_MODEL, 'checkpoints', 'model.pth'))

# Dimension of the question embedding vector.
QUESTION_EMBEDDING_DIM = 300
# Dimension of the relation embedding vector.
RELATION_EMBEDDING_DIM = 300
# Dimension of the question hidden vector.
QUESTION_HIDDEN_DIM = 200
# Dimension of the relation hidden vector.
RELATION_HIDDEN_DIM = 200

# Number of the training epochs.
EPOCH_NUM = 10
# Learning rate.
LEARNING_RATE = 0.0001
# The mini value that positive score bigger than negative.
MARGIN = 0.5


MAX_QUESTION_LENGTH = 20
MAX_RELATION_LEVEL_LENGTH = 2
MAX_WORD_LEVEL_LENGTH = 15

TRAIN_BATCH_SIZE = 512
DEV_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512

DEV_EVERY = 50
DEV_START_STEP = 0

PATIENCE = 30

QUESTION_NUMBER = 1639

USING_GRU = True
BIDIRECTIONAL = True

# layers number of RNN: (relation_num_layers, question_num_layers).
NUM_LAYERS = (2, 1)