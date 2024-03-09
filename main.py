import torch
from models import CharRNN
from my_utils import generate_training_json, find_latest_model, remove_old_models
from trainloop import train_rnn_model
from hyperparameters import HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE, LEARNING_RATE, DEVICE, NUM_EPOCHS
from datasets import train_dataloader, vocab_length
import time as time_module

model = CharRNN(input_size=vocab_length, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    print("Training the model")
    print("Using device:", DEVICE)

    model_file, current_epoch = find_latest_model()
    remove_old_models(current_epoch)
    start_time = time_module.time()

    if model_file:
        print(f"Latest untrained model found: {model_file}, epoch {current_epoch}. Continue training? (y/n)")
        answer = input()
        if answer.lower() == "y":
            model.load_state_dict(torch.load(model_file))
        else:
            current_epoch = 0
            generate_training_json()
            remove_old_models(NUM_EPOCHS+1) # Remove all models

    try:
        trained_model = train_rnn_model(model, loss_fn, optimizer, train_dataloader)
        torch.save(trained_model.state_dict(), f"rnn_model.pth")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
        torch.save(model.state_dict(), f"rnn-{current_epoch}.pth")