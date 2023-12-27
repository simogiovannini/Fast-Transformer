import os
from transformers import AutoTokenizer
from datasets import load_dataset


def batch_iterator(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def train_tokenizer(path, batch_size, vocabulary_size):
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    all_valid_texts = [txt for k in dataset.keys() for txt in dataset[k]['text'] if len(txt) > 0 and not txt.startswith(" =")]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenizer = tokenizer.train_new_from_iterator(batch_iterator(all_valid_texts, batch_size), vocab_size=vocabulary_size)
    tokenizer.save_pretrained(path)
    pass


def load_tokenizer(batch_size, vocabulary_size):
    path = './tokenizer/'
    if not os.path.exists(path):
        os.mkdir(path, mode=0o755)
        train_tokenizer(path, batch_size, vocabulary_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return tokenizer

