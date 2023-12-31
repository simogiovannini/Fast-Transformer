import os
import torch
from torch.utils.data import DataLoader
from bert import TokenPredictionHead
from dataset import load_dataset
from tokenizer import load_tokenizer
from torch.optim import Adam

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKENIZER_BATCH_SIZE = 256  # Batch-size to train the tokenizer on
TOKENIZER_VOCABULARY = 25000  # Total number of unique subwords the tokenizer can have

MAX_LENGTH = 128  # Maximum number of tokens in an input sample after padding

EMBEDDING_SIZE = 512

DROPOUT = 0.1

N_HEADS = 4

N_LAYERS = 8

D_FF = EMBEDDING_SIZE * 4

TRAIN_BATCH_SIZE = 8  # Batch-size for pretraining the model on
TRAIN_EPOCHS = 30  # Maximum number of epochs to train the model for
LEARNING_RATE = 1e-4  # Learning rate for training the model

tokenizer = load_tokenizer(TOKENIZER_BATCH_SIZE, TOKENIZER_VOCABULARY) 

dataset = load_dataset(f'./datasets/tiny_pretraining_dataset_{MAX_LENGTH}.pt')

train_loader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, pin_memory=True)

model = TokenPredictionHead(TOKENIZER_VOCABULARY, EMBEDDING_SIZE, MAX_LENGTH, N_LAYERS, N_HEADS, D_FF, DROPOUT, attention_type='clustered')
model.to(DEVICE)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

optim = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.NLLLoss(ignore_index=0)

for epoch in range(TRAIN_EPOCHS):

    avg_loss = 0.0
        
    for i, data in enumerate(train_loader):
        if i % 50 == 0:
            print(f'EP {epoch}: batch {i}/{len(train_loader)}')

        features, labels = data[0].to(DEVICE), data[1].to(DEVICE)

        output = model.forward(features)

        loss = criterion(output.transpose(1, 2), labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        avg_loss += loss.item()
        
    print(f'EP {epoch}: avg_loss={avg_loss / len(train_loader)}')

path = './models/'
if not os.path.exists(path):
    os.mkdir(path, mode=0o755)

torch.save(model.bert, 'models/pretrained.pt')