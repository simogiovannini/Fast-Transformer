import torch
from datasets import load_dataset
from glue_models import OneSentenceDataset, OneSentenceModel, TwoSentencesDataset, TwoSentencesModel
from tokenizer import load_tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader


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

TRAIN_BATCH_SIZE = 2
TRAIN_EPOCHS = 8

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

pretrained_model = torch.load(MODEL_PATH)

for task in GLUE_TASKS.keys():
    print(task)
    n_sentences_task = GLUE_TASKS[task][0]
    test_dataset_name = GLUE_TASKS[task][1]

    dataset = load_dataset('glue', task)

    if n_sentences_task == 1:
        train_dataset = OneSentenceDataset(dataset['train'], tokenizer)
        test_dataset = OneSentenceDataset(dataset[test_dataset_name], tokenizer)
        n_classes = train_dataset.get_n_classes()
        model = OneSentenceModel(pretrained_model, n_classes)
    elif n_sentences_task == 2:
        train_dataset = TwoSentencesDataset(dataset['train'], tokenizer)
        test_dataset = TwoSentencesDataset(dataset[test_dataset_name], tokenizer)
        n_classes = train_dataset.get_n_classes()
        model = TwoSentencesModel(pretrained_model, n_classes)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, pin_memory=True)

    model = model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = False

    # for param in model.fc.parameters():
        # param.requires_grad = True
    
    optimizer = Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(TRAIN_EPOCHS):
        avg_loss = 0
        
        for i, (X, y) in enumerate(train_dataloader):
            if n_sentences_task == 1:
                X = X.to(DEVICE)
            elif n_sentences_task == 2:
                X = (X[0].to(DEVICE), X[1].to(DEVICE))
            y = y.to(DEVICE)

            output = model(X)
            
            output = torch.nn.functional.softmax(output, 1)

            loss = criterion(output, y)
            avg_loss += loss
            print(output)
            print(loss.item())
            break
        break

    with torch.inference_mode():
        for i, (X, y) in enumerate(test_dataloader):
            if n_sentences_task == 1:
                X = X.to(DEVICE)
            elif n_sentences_task == 2:
                X = (X[0].to(DEVICE), X[1].to(DEVICE))
            y = y.to(DEVICE)

            output = model(X)
            
            output = torch.nn.functional.softmax(output, 1)
            pred = torch.argmax(output, dim=1, keepdim=True)
            
            # TODO: terminare calcolo accuratezza

            print(output)
            print(pred)
            break

    del model
    torch.cuda.empty_cache()