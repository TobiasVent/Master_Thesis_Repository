
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
# Original Normalisierungswerte
original_train_target_mean = -0.9489913
original_train_target_std = 0.81420016

# Daten laden
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

# PyTorch-Modell
class LSTMModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # letztes Zeitfenster
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = LSTMModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop mit EarlyStopping
patience = 10
best_loss = float("inf")
epochs_no_improve = 0
num_epochs = 50
train_losses = []
val_losses = []

plt.savefig("scatterplot_pytorch.png")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False)
    for xb, yb in train_loader:
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

# Bestes Modell laden
model.load_state_dict(best_model_state)

# Evaluation
model.eval()
predictions, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).squeeze().cpu().numpy()
        predictions.extend(preds)
        targets.extend(yb.squeeze().numpy())

predictions = np.array(predictions)
targets = np.array(targets)


plt.figure(figsize=(8,6))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot_pytorch.png")




# Inverse Normalisierung (Z-Transformation rückgängig machen)
predictions_rescaled = predictions * original_train_target_std + original_train_target_mean
targets_rescaled = targets * original_train_target_std + original_train_target_mean

# Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(targets_rescaled, predictions_rescaled, alpha=0.3, s=5)
plt.plot([targets_rescaled.min(), targets_rescaled.max()],
         [targets_rescaled.min(), targets_rescaled.max()],
         'r--', label='Perfekte Vorhersage')
plt.xlabel("Ground Truth")
plt.ylabel("Vorhersage")
plt.title("Scatterplot: Vorhersage vs. Ground Truth (PyTorch)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("scatterplot_pytorch.png")
plt.show()

# Modell speichern
torch.save(model.state_dict(), "/data/stu231428/Transformed_data_LSTM/lstm_model_pytorch.pt")
