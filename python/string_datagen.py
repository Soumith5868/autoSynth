import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader


#parameters
embedding_dim = 512
hidden_dim = 512
batch_size = 64
num_epochs = 20
learning_rate = 0.0001
vocab_size = len(cha)# use your particular charecter index dictionary

class StringGEN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super().__init__()
        """used GRU model for the following string generation so that we get higher accuracy rather than using LSTM model as the data is quite less"""
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        output, hidden = self.gru(x, hidden)
        logits = self.fc(output)
        return logits, hidden

#choose your particular device for your requirements
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X_tensor,Y_tensor = prepare_dataset(full_names, cha)
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = StringGEN(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=cha["<PAD>"])


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for bat_x, bat_y in dataloader:
        bat_x, bat_y = bat_x.to(device), bat_y.to(device)
        optimizer.zero_grad()
        logits, _ = model(bat_x)
        loss = loss_fn(logits.view(-1, vocab_size), bat_y.view(-1))
        loss.backward()
        optimizer.step()
        #scheduler.step()  # Optional if you're using a learning rate scheduler
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


