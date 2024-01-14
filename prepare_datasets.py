import os
from datasets import load_dataset
import random
from dataset import OneSentenceDataset, TwoSentencesDataset
from tokenizer import load_tokenizer
import torch


def tokenize_sentence(sentence, seq_len, tokenizer):
    tokens = tokenizer(sentence)
    tokens = tokens.input_ids[1:-1]
    if len(tokens) > (seq_len - 2):
            tokens = tokens[:seq_len - 2]
        
    tokens = [tokenizer.vocab['[CLS]']] + tokens + [tokenizer.vocab['[SEP]']]
    tokens = pad_sentence(tokens, tokenizer, seq_len)
    return torch.tensor(tokens)


def create_glue_dataset(path, seq_len, tokenizer, task_name):
    GLUE_TASKS = {'cola': [1, 'validation'],
              'mnli': [2, 'validation_matched'],
              'mrpc': [2, 'validation'],
              'qnli': [2, 'validation'],
              'qqp': [2, 'validation'],
              'rte': [2, 'validation'],
              'sst2': [1, 'validation'],
              'stsb': [2, 'validation'],
              'wnli': [2, 'validation']
              }
    
    print(f'Preparing {task_name} datasets...')
    
    n_sentences_task = GLUE_TASKS[task_name][0]
    test_dataset_name = GLUE_TASKS[task_name][1]

    dataset = load_dataset('glue', task_name)
    train_dataset = dataset['train']
    test_dataset = dataset[test_dataset_name]

    datasets = [train_dataset, test_dataset]
    
    if task_name == 'stsb':
        n_classes = 5
    else:
        n_classes = len(train_dataset.features['label'].names)
    
    column_names = [name for name in train_dataset[0].keys()]

    if n_sentences_task == 1:
        sentence_key, label_key = column_names[0], column_names[1]
        
        for i, d in enumerate(datasets):
            sentences = []
            labels = []

            for row in d:
                sentence = row[sentence_key]
                label = row[label_key]

                tokenized_sentence = tokenize_sentence(sentence, seq_len, tokenizer)

                sentences.append(tokenized_sentence)
                labels.append(torch.tensor(label))
            
            d_pair = (sentences, labels)
            d_pair = OneSentenceDataset(d_pair, n_classes)

            if i == 0:
                torch.save(d_pair, path + f'/{task_name}_train_dataset_{seq_len}.pt')
            else:
                torch.save(d_pair, path + f'/{task_name}_test_dataset_{seq_len}.pt')
        
    elif n_sentences_task == 2:
         sentence1_key, sentence2_key, label_key = column_names[0], column_names[1], column_names[2]

         for i, d in enumerate(datasets):
            first_sentences = []
            second_sentences = []
            labels = []

            for row in d:
                first_sentence = row[sentence1_key]
                second_sentence = row[sentence2_key]
                if task_name == 'stsb':
                    label = round(row[label_key])
                    if label == 0:
                        label = 1
                else:
                    label = row[label_key]
                tokenized_first_sentence = tokenize_sentence(first_sentence, seq_len, tokenizer)
                tokenized_second_sentence = tokenize_sentence(second_sentence, seq_len, tokenizer)

                first_sentences.append(tokenized_first_sentence)
                second_sentences.append(tokenized_second_sentence)
                labels.append(torch.tensor(label))
            
            d_triplet = (first_sentences, second_sentences, labels)
            d_triplet = TwoSentencesDataset(d_triplet, n_classes)

            if i == 0:
                torch.save(d_triplet, path + f'/{task_name}_train_dataset_{seq_len}.pt')
            else:
                torch.save(d_triplet, path + f'/{task_name}_test_dataset_{seq_len}.pt')
    
    print(f'Successfully saved {task_name} datasets!')
    pass


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
    print('Preparing pretraining dataset...')
    dataset = load_dataset('wikitext', 'wikitext-2-v1')
    all_valid_texts = [txt for k in dataset.keys() for txt in dataset[k]['text'] if len(txt) > 0 and not txt.startswith(" =")]
    all_valid_texts = [txt for _ in range(4) for txt in all_valid_texts]
    sentences = []
    masks = []
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
    
    dataset = (sentences, masks)
    dataset = OneSentenceDataset(dataset)
    torch.save(dataset, path + f'/pretraining_dataset_{seq_len}.pt')
    print('Successfully saved pretraining dataset!')
    pass


PATH = './datasets/'
SEQUENCE_LENGTH = 64
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_VOCABULARY = 25000

if not os.path.exists(PATH):
    os.mkdir(PATH)

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

# create_pretraining_dataset(PATH, SEQUENCE_LENGTH, tokenizer)

create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'cola')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'mnli')
create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'mrpc')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'qnli')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'qqp')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'rte')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'sst2')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'stsb')
# create_glue_dataset(PATH, SEQUENCE_LENGTH, tokenizer, 'wnli')