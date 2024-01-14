import torch
from bert import BERTModel
from glue_models import OneSentenceModel, TwoSentencesModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

GLUE_TASKS = {'cola': [1, 'validation'],
              #'mnli': [2, 'validation_matched'],
              'mrpc': [2, 'validation'],
              #'qnli': [2, 'validation'],
              #'qqp': [2, 'validation'],
              #'rte': [2, 'validation'],
              #'sst2': [1, 'validation'],
              #'stsb': [2, 'validation'],
              #'wnli': [2, 'validation']
              }

TOKENIZER_VOCABULARY = 25000

MAX_LENGTH = 64  # Maximum number of tokens in an input sample after padding

EMBEDDING_SIZE = 128

DROPOUT = 0.1

N_HEADS = 4

N_LAYERS = 2

D_FF = EMBEDDING_SIZE * 2

TRAIN_BATCH_SIZE = 8  # Batch-size for pretraining the model on
TRAIN_EPOCHS = 10  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

ATTENTION_TYPE = 'clustered'
N_CLUSTERS = 4

writer = SummaryWriter()

for task in GLUE_TASKS.keys():
    print(task)

    if ATTENTION_TYPE == 'full':
        tb_path_prefix = f'{ATTENTION_TYPE}/{task}'
    else:
        tb_path_prefix = f'{ATTENTION_TYPE}/{N_CLUSTERS}/{task}'

    n_sentences_task = GLUE_TASKS[task][0]

    train_dataset = torch.load(f'./datasets/{task}_train_dataset_{MAX_LENGTH}.pt')
    test_dataset = torch.load(f'./datasets/{task}_test_dataset_{MAX_LENGTH}.pt')
    

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, pin_memory=True)

    bert_model = BERTModel(TOKENIZER_VOCABULARY, EMBEDDING_SIZE, MAX_LENGTH, N_LAYERS, N_HEADS, D_FF, DROPOUT, attention_type=ATTENTION_TYPE, n_clusters=N_CLUSTERS)

    if n_sentences_task == 1:
        model = OneSentenceModel(bert_model, train_dataset.n_classes)
    else:
        model = TwoSentencesModel(bert_model, train_dataset.n_classes)

    model = model.to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = torch.nn.CrossEntropyLoss()
    
    for ep in range(TRAIN_EPOCHS):
        avg_loss = 0
        print(f'EP: {ep} / {TRAIN_EPOCHS}')
        
        for i, (X, y) in enumerate(train_dataloader):
            if i % 10 == 0:
                print(f'batch {i}/{len(train_dataloader)}')
                
            if n_sentences_task == 1:
                X = X.to(DEVICE)
            elif n_sentences_task == 2:
                X = (X[0].to(DEVICE), X[1].to(DEVICE))
            y = y.to(DEVICE)

            output = model(X)
            
            output = torch.nn.functional.softmax(output, 1)

            loss = criterion(output, y)
            avg_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Loss: {avg_loss}')
        
        writer.add_scalar(f'{tb_path_prefix}/Loss/train', avg_loss, ep)
            

    print('Now evaluating...')

    with torch.inference_mode():
        total_acc = 0
        for i, (X, y) in enumerate(test_dataloader):
            if n_sentences_task == 1:
                X = X.to(DEVICE)
            elif n_sentences_task == 2:
                X = (X[0].to(DEVICE), X[1].to(DEVICE))
            y = y.to(DEVICE)

            output = model(X)
            
            output = torch.nn.functional.softmax(output, 1)
            pred = torch.argmax(output, dim=1, keepdim=True).T

            curr_acc = (pred == y).sum().item() / pred.shape[1]
            total_acc += curr_acc
            
    total_acc /= len(test_dataloader)
    total_acc *= 100
    writer.add_scalar(f'{tb_path_prefix}/Accuracy/test', total_acc)
    print(f'Test accuracy on {task}: {total_acc}')
    del model
    torch.cuda.empty_cache()
    