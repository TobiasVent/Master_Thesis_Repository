
from tensorflow.keras.models import Sequential
import numpy as np
import pickle

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import shap
import matplotlib.pyplot as plt 



original_train_target_mean = -0.9489913

original_train_target_std = 0.81420016





feature_train = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1958_1988_feature.pkl",'rb'))
targets_train = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1958_1988_target.pkl",'rb'))
feature_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1988_2018_feature.pkl",'rb'))
targets_test = pickle.load(open("/data/stu231428/Transformed_data_LSTM/EQ_PACIFIC_1988_2018_target.pkl",'rb'))



# 1. Build samples
def create_samples(feature, targets, W):
    # feature: (num_coords, 720, 10), targets: (num_coords, 720,1)
    # W: window size
    X_train, y_train = [], []
    num_coords, T, _ = feature.shape
    for c in range(num_coords):
        for t in range(T - W ):
            X_train.append(feature[c, t:t+W])
            y_train.append(targets[c, t+W])
    return np.array(X_train), np.array(y_train)


window = 4
X_train, y_train = create_samples(feature_train, targets_train, window)
X_test, y_test = create_samples(feature_test, targets_test, window)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")







# 4. Model
model = Sequential([
    LSTM(64, input_shape=(window, 10)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile('adam', 'mse', metrics=['mae'])

# 5. Train
es = EarlyStopping(patience=10, restore_best_weights=True)


#X_train shape = (num_coords * (T - W), W, 10)
#y_train shape = (num_coords * (T - W), 1)

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=50, batch_size=32,
          callbacks=[es])

# 6. Evaluate
mse, mae = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")

#save model 
model.save('/data/stu231428/Transformed_data_LSTM/lstm_model.h5')

# # 5. SHAP: GradientExplainer
# background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
# # Modell einmal aufrufen, damit intern alles initialisiert ist


# # Explainer ohne explizite Inputs/Outputs erzeugen
# explainer = shap.GradientExplainer(model, background)

# sample_idx = 1
# X_sample = X_train[sample_idx:sample_idx+1]
# shap_values = explainer.shap_values(X_sample)

# # 6. SHAP-Heatmap: Zeit X_train Feature
# shap_arr = shap_values[0].squeeze()  # 

# plt.figure(figsize=(10, 6))
# plt.imshow(shap_arr.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
# plt.colorbar(label="SHAP value (impact on model output)")
# #plt.yticks(ticks=np.arange(len(.columns[1:])), labels=trainings_set.columns[1:])

# plt.xticks(ticks=np.arange(window), labels=[f"t-{window - i}" for i in range(window)])
# plt.title("SHAP Values Heatmap (Feature × Time Step)")
# plt.xlabel("Time Step")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig("shap_heatmap.png")
