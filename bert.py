from torch.utils.data import DataLoader
from dataset import BERTDataset, load_raw_sentences
from tokenizer import load_tokenizer

TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
TRAIN_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY)

sentences = load_raw_sentences()

dataset = BERTDataset(sentences, tokenizer, MAX_LENGTH)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

sample_data = next(iter(train_loader))

print(sample_data)