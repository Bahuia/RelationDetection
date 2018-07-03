# Config for project

# Path of the training data.
TRAIN_PATH = './data/train.json'
# Path of the test data.
TEST_PATH = './data/test.json'

# Dimension of the question embedding vector.
QUESTION_EMBEDDING_DIM = 300
# Dimension of the relation embedding vector.
RELATION_EMBEDDING_DIM = 300
# Dimension of the question hidden vector.
QUESTION_HIDDEN_DIM = 200
# Dimension of the relation hidden vector.
RELATION_HIDDEN_DIM = 200

# Number of the training epochs.
EPOCH_NUM = 30
# Learning rate.
LEARNING_RATE = 0.001
# The mini value that positive score bigger than negative.
MARGIN = 0.5

