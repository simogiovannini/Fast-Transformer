from datasets import load_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer
from transformers import BertTokenizer
import torch
import nltk
from torch.utils.data import DataLoader

from dataset import BERTDataset

def batch_iterator(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

BLOCK_SIZE = 128  # Maximum number of tokens in an input sample
NSP_PROB = 0.50  # Probability that the next sentence is the actual next sentence in NSP
SHORT_SEQ_PROB = 0.1  # Probability of generating shorter sequences to minimize the mismatch between pretraining and fine-tuning.
MAX_LENGTH = 512  # Maximum number of tokens in an input sample after padding

MLM_PROB = 0.2  # Probability with which tokens are masked in MLM

TRAIN_BATCH_SIZE = 2  # Batch-size for pretraining the model on
MAX_EPOCHS = 1  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

dataset = load_dataset('wikitext', 'wikitext-2-v1')

all_valid_texts = [txt for k in dataset.keys() for txt in dataset[k]['text'] if len(txt) > 0 and not txt.startswith(" =")]

print(len(all_valid_texts))

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(all_valid_texts, TOKENIZER_BATCH_SIZE), vocab_size=TOKENIZER_VOCABULARY
)

input = 'how are you?'

res = tokenizer(input)
encoding = tokenizer.encode(input)

print(res)
print(encoding)
print(tokenizer.convert_ids_to_tokens(encoding))

res = tokenizer(input, input, padding='max_length')
print(len(res['input_ids']))



nltk.download('punkt')
tok = nltk.data.load('tokenizers/punkt/english.pickle')


res1 = [s for sent in all_valid_texts for s in tok.tokenize(sent) if len(s) < 254]

print(tokenizer.vocab['[MASK]'])
print(tokenizer.vocab['[UNK]'])
print(tokenizer.vocab['[SEP]'])
print(tokenizer.vocab['[CLS]'])
print(tokenizer.vocab['[PAD]'])

print(len(tokenizer.vocab))

dataset = BERTDataset(res1, tokenizer, MAX_LENGTH)

train_loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
sample_data = next(iter(train_loader))

print(sample_data)