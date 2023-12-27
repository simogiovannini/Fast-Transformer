from torch.utils.data import Dataset
import torch
import random
import nltk
from datasets import load_dataset


def load_raw_sentences():
    nltk.download('punkt')
    tok = nltk.data.load('tokenizers/punkt/english.pickle')

    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    all_valid_texts = [txt for k in dataset.keys() for txt in dataset[k]['text'] if len(txt) > 0 and not txt.startswith(" =")]

    single_sentences = [s for sent in all_valid_texts for s in tok.tokenize(sent) if len(s) < 510]
    full_sentences = [sent for sent in all_valid_texts if len(sent) < 510]
    return single_sentences + full_sentences


class BERTDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=512):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.texts = texts
        self.n_texts = len(texts)

    def __len__(self):
        return self.n_texts

    def __getitem__(self, index):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next flag)
        sentence = self.texts[index]

        # Step 2: replace random words in sentence with mask / random words
        s_random, s_label = self.mask_and_replace_tokens(sentence)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        s_random = [self.tokenizer.vocab['[CLS]']] + s_random + [self.tokenizer.vocab['[SEP]']]
        s_label = [self.tokenizer.vocab['[PAD]']] + s_label + [self.tokenizer.vocab['[PAD]']]
        mask = [1 for _ in range(len(s_random))]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        pad_token = self.tokenizer.vocab['[PAD]']
        pad_len = self.seq_len - len(s_random)
        padding = [pad_token for _ in range(pad_len)]

        s_random.extend(padding)
        s_label.extend(padding)
        mask.extend(padding)

        output = {'bert_input': s_random,
                  'bert_label': s_label,
                  'attention_mask': mask,
                }

        return {key: torch.tensor(value) for key, value in output.items()}

    def mask_and_replace_tokens(self, sentence):
        tokens = self.tokenizer(sentence)['input_ids']
        tokens = tokens[1:-1]
        output_label = [0 for i in range(len(tokens))]

        # 15% of the tokens would be replaced
        for i in range(len(tokens)):
            prob = random.random()

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                        output_label[i] = tokens[i]
                        tokens[i] = self.tokenizer.vocab['[MASK]']

                # 10% chance change token to random token
                elif prob < 0.9:
                        output_label[i] = tokens[i]
                        tokens[i] = random.randrange(len(self.tokenizer.vocab))
        return tokens, output_label