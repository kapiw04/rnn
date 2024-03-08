import torch
from my_utils import char_tensor, idx_to_char, tensor_to_str
from datasets import vocab
from hyperparameters import HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE
import sys

tokens_to_generate = 1000
sys.setrecursionlimit(tokens_to_generate + 1000)

def predict_next_char(model, input_tensor, temperature=1.0):
    input_tensor.unsqueeze_(0)
    prediction = model(input_tensor)
    prediction = prediction.squeeze(0)
    new_input_sampled = torch.multinomial(torch.exp(prediction / temperature), 1) 
    new_one_hot = torch.zeros(1, 1, len(vocab))
    new_one_hot[0, 0, new_input_sampled] = 1
    
    return new_input_sampled.item()

def generate_text(model, start_string, temperature=1.0):
    """
        Recursively generates text from the model.
    """
    global tokens_to_generate
    if tokens_to_generate < 1 and start_string[-1] in ['.', '!', '?']:
        return start_string
    # Set the model to evaluation mode

    # Convert the start string to a tensor
    model.eval()

    input_tensor = char_tensor(start_string, vocab)

    # Get the next character
    next_char = predict_next_char(model, input_tensor, temperature)
    tokens_to_generate -= 1
    return generate_text(model, start_string + idx_to_char(next_char, vocab), temperature) 

if __name__ == "__main__":
    from models import CharRNN
    model = CharRNN(input_size=len(vocab), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    # model load on cpu
    model.load_state_dict(torch.load("rnn.pth", map_location=torch.device('cpu')))
    print(generate_text(model, "WARWICK:", temperature=0.5))