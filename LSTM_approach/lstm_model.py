import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import shap
import matplotlib.pyplot as plt 
feature = pickle.load(open("EQ_PACIFIC_feature.pkl",'rb'))
targets = pickle.load(open("EQ_PACIFIC_target.pkl",'rb'))



# 1. Build samples
def create_samples(feature, targets, W):
    # feature: (num_coords, 720, 10), targets: (num_coords, 720,1)
    # W: window size
    X, y = [], []
    num_coords, T, _ = feature.shape
    for c in range(100):
        for t in range(T - W ):
            X.append(feature[c, t:t+W])
            y.append(targets[c, t+W])
    return np.array(X), np.array(y)


window = 12 
X, y = create_samples(feature, targets, window)
print(f"X shape: {X.shape}, y shape: {y.shape}")


# 2. Split
split1 = int(0.80 * X.shape[0])   
  
X_train,  X_test = X[:split1], X[split1:]
y_train, y_test = y[:split1], y[split1:]


# 3. Normalize
scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
#X_val   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# 4. Model
model = Sequential([
    LSTM(64, input_shape=(window, 10)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile('adam', 'mse', metrics=['mae'])

# 5. Train
es = EarlyStopping(patience=5, restore_best_weights=True)


#X_train shape = (num_coords * (T - W), W, 10)
#y_train shape = (num_coords * (T - W), 1)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50, batch_size=64,
          callbacks=[es])

# 6. Evaluate
mse, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")



# 5. SHAP: GradientExplainer
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
# Modell einmal aufrufen, damit intern alles initialisiert ist


# Explainer ohne explizite Inputs/Outputs erzeugen
explainer = shap.GradientExplainer(model, background)

sample_idx = 1
X_sample = X_train[sample_idx:sample_idx+1]
shap_values = explainer.shap_values(X_sample)

# 6. SHAP-Heatmap: Zeit x Feature
shap_arr = shap_values[0].squeeze()  # 

plt.figure(figsize=(10, 6))
plt.imshow(shap_arr.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
plt.colorbar(label="SHAP value (impact on model output)")
#plt.yticks(ticks=np.arange(len(.columns[1:])), labels=trainings_set.columns[1:])

plt.xticks(ticks=np.arange(window), labels=[f"t-{window - i}" for i in range(window)])
plt.title("SHAP Values Heatmap (Feature × Time Step)")
plt.xlabel("Time Step")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("shap_heatmap.png")
