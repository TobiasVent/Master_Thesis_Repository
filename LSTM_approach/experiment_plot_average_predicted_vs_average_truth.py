import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load feature and target data
feature_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/NORTH_ATLANTIC_2006_2018_feature.pkl", 'rb'))
targets_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/NORTH_ATLANTIC_2006_2018_target.pkl", 'rb'))

print(f"feature_test shape: {feature_test.shape}, targets_test shape: {targets_test.shape}")

# Mean and std from original training data (used for inverse normalization)
original_train_target_mean = -0.9489913
original_train_target_std = 0.81420016

# Create time-windowed samples from the data
def create_samples(feature, targets, W):
    """
    Args:
        feature: array of shape (num_coords, T, num_features)
        targets: array of shape (num_coords, T, 1)
        W: window size (int)

    Returns:
        X: shape (num_coords*(T-W), W, num_features)
        y: shape (num_coords*(T-W), 1)
    """
    X_train, y_train = [], []
    num_coords, T, _ = feature.shape
    for c in range(num_coords):
        for t in range(T - W):
            X_train.append(feature[c, t:t+W])
            y_train.append(targets[c, t+W])
    return np.array(X_train), np.array(y_train)

# Set window size and create dataset
window = 3
X_test, y_test = create_samples(feature_test, targets_test, window)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Prepare PyTorch DataLoader
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
test_loader = DataLoader(test_dataset, batch_size=64)

# Model configuration
hidden_dim = 256
num_layers = 3
dropout = 0.32383408056252133
lr = 0.0001
weight_decay = 1.978060585506455e-05

# Define LSTM model (architecture must match the saved model)
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)            # out: (batch_size, seq_len, hidden_dim)
        x = out[:, -1, :]                # get output from the last time step
        x = self.dropout(x)
        x = self.fc1(x)                  # final prediction
        return x

# Load model and weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
model.load_state_dict(torch.load("/home/stu231428/trained_models/lstm_model.pt", map_location=device))
model.eval()

# Run inference on test data
predictions, targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).squeeze().cpu().numpy()
        predictions.extend(preds)
        targets.extend(yb.squeeze().numpy())

# Convert to NumPy arrays
predictions = np.array(predictions)
targets = np.array(targets)

# Inverse normalization of predictions and targets
predictions_rescaled = predictions * original_train_target_std + original_train_target_mean
targets_rescaled = targets * original_train_target_std + original_train_target_mean

# Reshape from flat list of samples to (num_coords, time_steps)
# Note: time_steps = original_time - window = 144 - window = 141
preds_2d = predictions_rescaled.reshape(72758, 141)
targets_2d = targets_rescaled.reshape(72758, 141)

# Calculate the mean across all coordinates for each time step
mean_preds = preds_2d.mean(axis=0)
mean_targets = targets_2d.mean(axis=0)

# Create date range for x-axis
start_date = pd.to_datetime("2006-01-01") + pd.DateOffset(months=window)
dates = pd.date_range(start=start_date, periods=mean_preds.shape[0], freq='M')
x_labels = dates.strftime("%b %Y")
xticks_to_show = np.arange(0, len(x_labels), 6)  # show every 6th month label

# Plot predicted vs. true mean values over time
plt.figure(figsize=(12, 6))
plt.plot(x_labels, mean_targets, label="True Mean")
plt.plot(x_labels, mean_preds, label="Predicted Mean")
plt.xlabel("Time")
plt.ylabel("Target Value")
plt.title("Predicted vs True Mean Target Values Over Time (Window Size 3)")
plt.legend()
plt.xticks(xticks_to_show, [x_labels[i] for i in xticks_to_show], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_target_vs_predicted_window_3.png")

# Compute and print Mean Squared Error between predicted and true means
mse = mean_squared_error(mean_targets, mean_preds)
print(f"Mean Squared Error over time averages: {mse:.4f}")


