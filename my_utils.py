import time
import sys
import torch

time_start = time.time()

def char_to_idx(char, vocab):
    return vocab.index(char)
    

def idx_to_char(idx, vocab):
    return vocab[idx]

def format_eta(seconds, include_hours=False):
    """
    Formats the ETA in a human-readable form.

    Args:
        seconds (float): The estimated time remaining in seconds.
        include_hours (bool): Whether to include hours in the formatted string.

    Returns:
        str: The formatted ETA.
    """
    if include_hours:
        return time.strftime('%Hh %Mm %Ss', time.gmtime(seconds))
    else:
        return time.strftime('%Mm %Ss', time.gmtime(seconds))

def calculate_eta(time_start, i, total_batches, epochs_remaining):
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
    avg_time_per_batch = (time_end - time_start) / (i + 1)
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
    eta_seconds = calculate_eta(time_start, i, total_batches, epochs_remaining)
    include_hours = eta_seconds >= 3600
    time_formatted = format_eta(eta_seconds, include_hours=include_hours)
    
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