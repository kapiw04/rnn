import torch
from my_utils import eta, save_model, load_training_from_json, update_training_json
from hyperparameters import DEVICE, NUM_EPOCHS, LEARNING_RATE_DECAY, AFTER_N_EPOCHS, LEARNING_RATE
from torch.optim import lr_scheduler
import os
import time

starting_epoch = 0

def train_one_epoch(model, loss_fn, optimizer, dataloader: torch.utils.data.DataLoader, epoch: int):
    """
        model: The RNN model
        loss_fn: The loss function
        optimizer: The optimizer
        dataset: The dataset
        epoch: The current epoch
    """
    start_time = time.time()
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

    training_data = load_training_from_json('current_model.json')
    total_time = time.time() - start_time
    new_time_elapsed = total_time + training_data['total_time_elapsed']
    update_training_json("current_model.json", key='total_time_elapsed', value=new_time_elapsed)
    update_training_json("current_model.json", key='epoch', value=epoch+1) 
    return running_loss / len(dataloader)

def train_rnn_model(model, loss_fn, optimizer, dataloader: torch.utils.data.DataLoader):
    """
        model: The RNN model
        loss_fn: The loss function
        optimizer: The optimizer
        dataset: The dataset
    """
    training_data = load_training_from_json('current_model.json')
    starting_epoch = training_data['epoch']
    
    model = model.to(DEVICE)
    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=AFTER_N_EPOCHS, gamma=LEARNING_RATE_DECAY, last_epoch=starting_epoch-1)
    model.train()
    for epoch in range(starting_epoch, NUM_EPOCHS):
        loss = train_one_epoch(model, loss_fn, optimizer, dataloader, epoch)
        print(f'\rEpoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss}')
        scheduler.step()    
        save_model(model, f'rnn-{epoch+1}.pth')
        if epoch > 0:
            os.remove(f'rnn-{epoch}.pth')

    return model