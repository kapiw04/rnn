import torch
from my_utils import char_tensor
from datasets import vocab
from hyperparameters import HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE

def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    # Convert the start string to tensor
    input_eval = char_tensor(start_string, vocab).unsqueeze(0)
    # Empty string to store the generated textt
    text_generated = []

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for i in range(num_generate):
            output = model(input_eval)
            # Apply temperature to the output probabilities
            output = output.squeeze().div(temperature).exp()
            # Sample the next character from the output probabilities
            predicted_id = torch.multinomial(output, num_samples=1).item()
            # Pass the predicted character as the next input to the model
            input_eval.fill_(predicted_id)
            # Convert the predicted character index to a character and append it to the generated text
            text_generated.append(vocab[predicted_id])


    return start_string + ''.join(text_generated)

if __name__ == "__main__":
    from models import CharRNN
    model = CharRNN(input_size=len(vocab), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load('rnn_model.pth'))
    print(generate_text(model, "ROMEO: ", num_generate=1000, temperature=0.5))