from torch.utils.data import DataLoader
from bert import BERTModel
from dataset import BERTDataset, load_raw_sentences
from tokenizer import load_tokenizer

TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

EMBEDDING_SIZE = 768

DROPOUT = 0.1

N_HEADS = 12

N_LAYERS = 12

D_FF = 768 * 4

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
TRAIN_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

sentences = load_raw_sentences()

dataset = BERTDataset(sentences, tokenizer, MAX_LENGTH)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

sample_data = next(iter(train_loader))

model = BERTModel(TOKENIZER_VOCABULARY, EMBEDDING_SIZE, MAX_LENGTH, N_LAYERS, N_HEADS, D_FF, DROPOUT)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

bert_input, bert_label, attention_mask = sample_data['bert_input'], sample_data['bert_label'], sample_data['attention_mask']

res = model(bert_input, attention_mask)

print(bert_input)
print(res)
print(bert_input.shape)
print(res.shape)