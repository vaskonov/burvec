import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class BiLSTMPOSTagger(nn.Module):

    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, text):

        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        
#         embedded = self.embedding(text)
#         outputs, (hidden, cell) = self.lstm(embedded)
#         predictions = self.fc(outputs)
        
        return predictions

    
# def tag_percentage(tag_counts):
#     total_count = sum([count for tag, count in tag_counts])
#     tag_counts_percentages = [(tag, count, count/total_count) for tag, count in tag_counts]
#     return tag_counts_percentages


def init_weights(m):
    for name, param in m.named_parameters():
        torch.manual_seed(1234)
        nn.init.normal_(param.data, mean = 0, std = 0.1)
        
        
def count_parameters(model):
    return (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))


# def categorical_accuracy(preds, y, tag_pad_idx):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
#     """
#     max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
#     non_pad_elements = (y != tag_pad_idx).nonzero()
#     correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
#     return correct.sum() / y[non_pad_elements].shape[0]


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        text = batch.text
        tags = batch.udtags
        
        optimizer.zero_grad()
        
        predictions = model(text)
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        loss = criterion(predictions, tags)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    preds = []
    golds = []
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.text
            tags = batch.udtags
            
            predictions = model(text)
            
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            
            max_preds = predictions.argmax(dim = 1, keepdim = True) # get the index of the max probability
            
            preds.extend([a[0] for a in max_preds.tolist()])
            golds.extend(tags.tolist())
            
            epoch_loss += loss.item()
    

    f1 = f1_score(golds, preds, average='weighted')
#     f1 = f1_score(golds, preds, average='macro')
    return epoch_loss / len(iterator), f1# epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs