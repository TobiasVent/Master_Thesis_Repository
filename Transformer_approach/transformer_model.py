import torch
import torch.nn as nn
import math
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
feature_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1958_1988_feature.pkl",'rb'))
targets_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1958_1988_target.pkl",'rb'))
feature_train = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1988_2018_feature.pkl",'rb'))
targets_train = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1988_2018_target.pkl",'rb'))

# feature: (num_coords, 720, 10), targets: (num_coords, 720,1)


# Samples erzeugen
def create_samples(feature, targets, W):
    X, y = [], []
    num_coords, T, _ = feature.shape
    for c in range(1000):
        for t in range(T - W):
            X.append(feature[c, t:t+W])
            y.append(targets[c, t+W])
    return np.array(X), np.array(y)

window = 4
X_train, y_train = create_samples(feature_train, targets_train, window)
X_test, y_test = create_samples(feature_test, targets_test, window)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

#X_train shape = (num_coords * (360 - W), W, 10)
#y_train shape = (num_coords * (360 - W), 1)

# Torch-Datasets
batch_size = 128
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)





class InputEmbedding(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        return x




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Erzeuge Matrix mit shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Sinus & Cosinus-Kodierung
        pe[:, 0::2] = torch.sin(position * div_term)  # gerade Positionen
        pe[:, 1::2] = torch.cos(position * div_term)  # ungerade Positionen

        pe = pe.unsqueeze(0)  # -> shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
    


class Transformer_Prediction(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=1, num_layers=1, d_model=512):
        super(Transformer_Prediction, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), 
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        src = self.positional_encoding(src)  # Hinzufügen der Positionskodierung
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        out = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        out = self.fc_out(out)  # (batch_size, seq_len, output_dim)
        return out



# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = Transformer_Prediction().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop mit EarlyStopping
patience = 10
best_loss = float("inf")
epochs_no_improve = 0
num_epochs = 50
train_losses = []
val_losses = []



for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for xb, yb in train_loader_tqdm:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb).squeeze()
            loss = criterion(outputs, yb.squeeze())
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break