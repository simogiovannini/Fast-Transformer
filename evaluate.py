import torch
from datasets import load_dataset
from glue_models import OneSentenceDataset, OneSentenceModel, TwoSentencesModel
from tokenizer import load_tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_NAME = 'pretrained'
MODEL_PATH = './models/' + MODEL_NAME + '.pt'

TOKENIZER_BATCH_SIZE = 256
TOKENIZER_VOCABULARY = 25000

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

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

pretrained_model = torch.load(MODEL_PATH)

for task in GLUE_TASKS.keys():
    print(task)
    dataset = load_dataset('glue', task)['train']
    dataset = OneSentenceDataset(dataset, tokenizer)
    print(len(dataset[0][0]))
    break