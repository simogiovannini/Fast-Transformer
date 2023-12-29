import os
from datasets import load_dataset
import random
from dataset import OneSentenceDataset
from tokenizer import load_tokenizer
import torch


def pad_sentence(tokens, tokenizer, seq_len):
    pad_token = tokenizer.vocab['[PAD]']
    pad_len = seq_len - len(tokens)
    padding = [pad_token for _ in range(pad_len)]
    tokens.extend(padding)
    return tokens

     
def mask_and_replace_tokens(sentence, tokenizer):
        tokens = tokenizer(sentence)['input_ids']
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
                        tokens[i] = tokenizer.vocab['[MASK]']

                # 10% chance change token to random token
                elif prob < 0.9:
                        output_label[i] = tokens[i]
                        tokens[i] = random.randrange(len(tokenizer.vocab))
        return tokens, output_label


def create_pretraining_dataset(path, seq_len, tokenizer):
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    all_valid_texts = [txt for k in dataset.keys() for txt in dataset[k]['text'] if len(txt) > 0 and not txt.startswith(" =")]
    all_valid_texts = [txt for _ in range(4) for txt in all_valid_texts]
    print(len(all_valid_texts))
    sentences = []
    masks = []
    i = 0
    for txt in all_valid_texts:
        tokenized_txt, masked_tokens = mask_and_replace_tokens(txt, tokenizer)
        
        if len(tokenized_txt) > (seq_len - 2):
            tokenized_txt = tokenized_txt[:seq_len - 2]
            masked_tokens = masked_tokens[:seq_len - 2]
        
        tokenized_txt = [tokenizer.vocab['[CLS]']] + tokenized_txt + [tokenizer.vocab['[SEP]']]
        masked_tokens = [tokenizer.vocab['[PAD]']] + masked_tokens + [tokenizer.vocab['[PAD]']]

        tokenized_txt = pad_sentence(tokenized_txt, tokenizer, seq_len)
        masked_tokens = pad_sentence(masked_tokens, tokenizer, seq_len)

        tokenized_txt = torch.tensor(tokenized_txt)
        masked_tokens = torch.tensor(masked_tokens)
        
        sentences.append(tokenized_txt)
        masks.append(masked_tokens)
        i += 1
        print(i)
    
    dataset = (sentences, masks)
    dataset = OneSentenceDataset(dataset)
    torch.save(dataset, PATH + '/pretraining_dataset.pt')
    pass


PATH = './datasets/'
SEQUENCE_LENGTH = 512
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_VOCABULARY = 25000

if not os.path.exists(PATH):
    os.mkdir(PATH)

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

create_pretraining_dataset(PATH, SEQUENCE_LENGTH, tokenizer)