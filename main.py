import torch
from models import CharRNN
from trainloop import train_rnn_model
from hyperparameters import HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE, LEARNING_RATE, DEVICE
from datasets import train_dataloader, vocab_length

model = CharRNN(input_size=vocab_length, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_rate=DROPOUT_RATE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    trained_model = train_rnn_model(model, loss_fn, optimizer, train_dataloader)
    torch.save(trained_model.state_dict(), "rnn_model.pth")
