import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm

def train(
    train_dataloader, 
    val_dataloader,
    model, 
    vocab_tox=None,
    vocab_detox=None,
    optimizer=None, 
    criterion=None, 
    learning_rate=1e-3,
    epochs=15
):
    
    """
        Train the given model
            :param train_dataloader: dataloader for train
            :param val_dataloader: dataloader for validation
            :param model: model to train
            :param vocab_tox: vocabulary for toxic text
            :param vocab_detox: vocabulary for translated text
            :param optimzier: optimizer for model,     default = Adam
            :param criterion: loss function for model, default = CrossEntropyLoss
            :param learding_rate: learning rate for optimizer and criterion, default = 1e-3
            :param epochs: number of epochs to train model: defualt = 15
    """
    
    optimizer = optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tox['<pad>'])
    
    total = 0
    loss_total = 0
    best = float('inf')
    loss_train_list = []
    loss_val_list = []
    for epoch in range(1, epochs + 1):
        loss_train = train_epoch(epoch, train_dataloader, seq2seq, optimizer, criterion)
        best, loss_val = val_epoch(epoch, val_dataloader, seq2seq, criterion, best_so_far=best)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)
    
    return loss_train_list, loss_val_list


def train_epoch(epoch, dataloader, seq2seq, optimizer, criterion):

    total_loss = 0
    total = 0
    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch}: train",
        leave=True,
    )
    
    seq2seq.train()
    for i, batch in loop:
        input, target = batch

        optimizer.zero_grad()
        
        outputs = seq2seq(input, target)
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            target.view(-1)
        )
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * input.shape[0]
        total += input.shape[0]
        loop.set_postfix({"loss": total_loss/total})

    return total_loss / total


def val_epoch(epoch, dataloader, seq2seq,
          criterion, best_so_far=0.0, seq2seq_path='seq2seq.pt'):
    
    total_loss = 0
    total = 0
    loop = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc=f"Epoch {epoch}: val",
        leave=True,
    )
    
    with torch.no_grad():
        seq2seq.eval()
        for i, batch in loop:
            input, target = batch

            outputs = seq2seq(input, target)
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                target.view(-1)
            )

            total_loss += loss.item() * input.shape[0]
            total += input.shape[0]
            loop.set_postfix({"loss": total_loss/total})

        Loss = total_loss / total
        if Loss < best_so_far:
            torch.save(seq2seq.state_dict(), seq2seq_path)
            return Loss, total_loss / total

    return best_so_far, total_loss / total