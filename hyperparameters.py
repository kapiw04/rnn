import torch

SEQUENCE_LEGTH = 100
HIDDEN_SIZE = 256
DROPOUT_RATE = 0.5
NUM_LAYERS = 2
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
MOMENTUM = 0.9

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")