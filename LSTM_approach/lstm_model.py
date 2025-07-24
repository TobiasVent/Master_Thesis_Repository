import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd



class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        return x
        #x = self.relu(self.fc1(x))
        #return self.fc2(x)






class LSTMSampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        X = torch.tensor(sample.features, dtype=torch.float32)
        y = torch.tensor(sample.target, dtype=torch.float32)
        meta = sample.meta
        return X, y, meta




# Daten laden
with open("/data/stu231428/Master_Thesis/Data/global_sample_1958_1988_0.005.pkl", "rb") as f:
    train_samples = pickle.load(f)


with open("/data/stu231428/Master_Thesis/Data/global_sample_1958_1988_0.005.pkl", "rb") as f:
    test_samples = pickle.load(f)


train_dataset = LSTMSampleDataset(train_samples)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)



test_dataset = LSTMSampleDataset(test_samples)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)



# Torch-Datasets
batch_size = 64



# PyTorch-Modell
hidden_dim = 256
num_layers = 3
dropout = 0.32383408056252133
lr = 0.0001
weight_decay = 1.978060585506455e-05


# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Training loop mit EarlyStopping
patience = 10
best_loss = float("inf")
epochs_no_improve = 0
num_epochs = 50
train_losses = []
val_losses = []



for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    
    for xb, yb, meta in train_loader_tqdm:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb).squeeze()
        loss = criterion(outputs, yb.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

        train_loader_tqdm.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    val_loss = 0
    model.eval()
    with torch.no_grad():
        for xb, yb, meta in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb).squeeze()
            loss = criterion(outputs, yb.squeeze())
            val_loss += loss.item() * xb.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


# Bestes Modell laden
model.load_state_dict(best_model_state)
#save best model
torch.save(model.state_dict(), "/home/stu231428/trained_models/lstm_model.pt")
# Evaluation
# model.eval()
# predictions, targets = [], []
# with torch.no_grad():
#     for xb, yb in test_loader:
#         xb = xb.to(device)
#         preds = model(xb).squeeze().cpu().numpy()
#         predictions.extend(preds)
#         targets.extend(yb.squeeze().numpy())

# predictions = np.array(predictions)
# targets = np.array(targets)


plt.figure(figsize=(8,6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot_pytorch.png")




# # Inverse Normalisierung (Z-Transformation rückgängig machen)
# predictions_rescaled = predictions * original_train_target_std + original_train_target_mean
# targets_rescaled = targets * original_train_target_std + original_train_target_mean

# # Scatterplot
# plt.figure(figsize=(8, 6))
# plt.scatter(targets_rescaled, predictions_rescaled, alpha=0.3, s=5)
# plt.plot([targets_rescaled.min(), targets_rescaled.max()],
#          [targets_rescaled.min(), targets_rescaled.max()],
#          'r--', label='Perfekte Vorhersage')
# plt.xlabel("Ground Truth")
# plt.ylabel("Vorhersage")
# plt.title("Scatterplot: Vorhersage vs. Ground Truth (PyTorch)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("scatterplot_pytorch.png")
# plt.show()

# # Modell speichern
# #torch.save(model.state_dict(), "/data/stu231428/Transformed_data_LSTM/lstm_model_pytorch.pt")
