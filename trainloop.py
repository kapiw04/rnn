import torch
from my_utils import eta
from hyperparameters import DEVICE, NUM_EPOCHS, LEARNING_RATE_DECAY, AFTER_N_EPOCHS
from torch.optim import lr_scheduler


def train_one_epoch(model, loss_fn, optimizer, dataloader: torch.utils.data.DataLoader, epoch: int):
    """
        model: The RNN model
        loss_fn: The loss function
        optimizer: The optimizer
        dataset: The dataset
        epoch: The current epoch
    """
    running_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f'\r{i}/{len(dataloader)}, Loss: {loss}', end='')
        eta(epoch, i, NUM_EPOCHS, dataloader)
    return running_loss / len(dataloader)

def train_rnn_model(model, loss_fn, optimizer, dataloader: torch.utils.data.DataLoader):
    """
        model: The RNN model
        loss_fn: The loss function
        optimizer: The optimizer
        dataset: The dataset
    """
    model = model.to(DEVICE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=AFTER_N_EPOCHS, gamma=LEARNING_RATE_DECAY)
    model.train()
    for epoch in range(NUM_EPOCHS):
        loss = train_one_epoch(model, loss_fn, optimizer, dataloader, epoch)
        print(f'\rEpoch: {epoch + 1}/{NUM_EPOCHS}, Loss: {loss}')
        scheduler.step()    

    return model