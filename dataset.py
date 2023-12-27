from torch.utils.data import Dataset
import torch
import random

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
        s1, s2, is_next = self.get_sentences(index)

        # Step 2: replace random words in sentence with mask / random words
        s1_random, s1_label = self.mask_and_replace_tokens(s1)
        s2_random, s2_label = self.mask_and_replace_tokens(s2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        s1_random = [self.tokenizer.vocab['[CLS]']] + s1_random + [self.tokenizer.vocab['[SEP]']]
        s2_random = s2_random + [self.tokenizer.vocab['[SEP]']]
        s1_label = [self.tokenizer.vocab['[PAD]']] + s1_label + [self.tokenizer.vocab['[PAD]']]
        s2_label = s2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(s1_random))] + [2 for _ in range(len(s2_random))])
        bert_input = s1_random + s2_random
        bert_label = s1_label + s2_label

        pad_token = self.tokenizer.vocab['[PAD]']
        pad_len = self.seq_len - len(bert_input)
        padding = [pad_token for _ in range(pad_len)]

        bert_input.extend(padding)
        bert_label.extend(padding)
        segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next}

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

    def get_sentences(self, index):
        s1 = self.texts[index]
        if random.random() > 0.5:
            if index < self.n_texts - 1:
                s2 = self.texts[index + 1]
            else:
                s2 = self.texts[index]
            return s1, s2, 1
        else:
            s2 = self.texts[random.randint(0, self.n_texts)]
            return s1, s2, 0