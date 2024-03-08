import time
import sys
import torch

time_start = time.time()

def char_to_idx(char, vocab):
    return vocab.index(char)
    

def idx_to_char(idx, vocab):
    return vocab[idx]

def format_eta(seconds):
    """
    Formats the ETA in a human-readable form.

    Args:
        seconds (float): The estimated time remaining in seconds.
        include_hours (bool): Whether to include hours in the formatted string.

    Returns:
        str: The formatted ETA.
    """
    include_hours = seconds >= 3600
    include_days = seconds >= 86400
    if include_days:
        return time.strftime('%jd %Hh %Mm %Ss', time.gmtime(seconds))
    if include_hours:
        return time.strftime('%Hh %Mm %Ss', time.gmtime(seconds))
    else:
        return time.strftime('%Mm %Ss', time.gmtime(seconds))

def calculate_eta(time_start, i, total_batches, epochs_remaining, epoch):
    """
    Calculates the estimated time of arrival (completion).

    Args:
        time_start (float): The start time of the operation.
        i (int): The current batch index.
        total_batches (int): The total number of batches.
        epochs_remaining (int): The number of epochs remaining.

    Returns:
        float: The ETA in seconds.
    """
    time_end = time.time()
    avg_time_per_batch = (time_end - time_start) / (i + 1 + (epoch * total_batches))
    time_to_finish_current_epoch = avg_time_per_batch * (total_batches - (i + 1))
    time_for_remaining_epochs = avg_time_per_batch * total_batches * epochs_remaining
    return time_to_finish_current_epoch + time_for_remaining_epochs

def eta(epoch, i, epochs, dataloader):
    """
    Prints the estimated time of arrival (completion) for the training process.

    Args:
        epoch (int): The current epoch.
        i (int): The current batch index within the epoch.
        epochs (int): The total number of epochs.
        dataloader (DataLoader): The DataLoader object being used.
        include_hours (bool): Whether to include hours in the formatted ETA.
    """
    global time_start
    epochs_remaining = epochs - epoch - 1
    total_batches = len(dataloader)
    eta_seconds = calculate_eta(time_start, i, total_batches, epochs_remaining, epoch)
    time_formatted = format_eta(eta_seconds)
    
    sys.stdout.write(f'\rEpoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{total_batches}, ETA: {time_formatted}')
    sys.stdout.flush()


def char_tensor(string, vocab):
    """
    Converts a string to a tensor of character indices.

    Args:
        string (str): The input string.
        vocab (str): The character vocabulary.

    Returns:
        torch.Tensor: The tensor of character indices.
    """
    tensor = torch.zeros(len(string), len(vocab), dtype=torch.float32)
    for i, char in enumerate(string):
        tensor[i][char_to_idx(char, vocab)] = 1
    return tensor


def save_model(model, path):
    """
    Saves the model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The file path to save the model to.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Loads the model from a file.

    Args:
        model (torch.nn.Module): The model to load the parameters into.
        path (str): The file path to load the model from.
    """
    model.load_state_dict(torch.load(path))


def tensor_to_str(tensor, vocab):
    """
    Converts a tensor of character indices to a string.

    Args:
        tensor (torch.Tensor): The tensor of character indices.

    Returns:
        str: The string representation of the tensor.
    """
    # torch.multinomial(torch.exp(prediction / temperature), 1) 

    return ''.join([idx_to_char(torch.argmax(tensor[i]).item(), vocab) for i in range(tensor.shape[0])])