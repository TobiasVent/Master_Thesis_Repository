import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import namedtuple



# Load test dataset
print("Loading and sorting  dataset...")
file_path = f'/data/stu231428/Master_Thesis/Data/global_sample_1958_1998_0.05.pkl'
df = pd.read_pickle(file_path)
df = df.sort_values(by=["coord_id", "time_counter"])

# Define feature columns
feature_columns = ['SST', 'SAL', 'ice_frac', 'mixed_layer_depth', 'heat_flux_down',
                   'water_flux_up', 'stress_X', 'stress_Y', 'currents_X', 'currents_Y']

# Load normalization statistics from training
print("Loading normalization stats from training data...")
norm_stats_path = "/data/stu231428/Transformed_data_LSTM/normalization_stats_global_0.01.pkl"
with open(norm_stats_path, "rb") as f:
    norm_stats = pickle.load(f)

feature_means = pd.Series(norm_stats['mean'])
feature_stds = pd.Series(norm_stats['std'])

# Apply z-normalization using training statistics
print("Applying z-normalization using training statistics...")
df[feature_columns] = (df[feature_columns] - feature_means) / feature_stds

# Create data structures
window_size = 4
features = []
targets = []
meta = []
all_samples = []
Sample = namedtuple("Sample", ["features", "target", "meta"])
# Group by location and apply sliding window with progress bar
print("Generating features and targets from  data...")
grouped = df.groupby(['nav_lat', 'nav_lon'])

for (lat, lon), group in tqdm(grouped, desc="Processing  locations"):
    group = group.reset_index(drop=True)
    for i in range(window_size - 1, len(group)):
        window = group.iloc[i - window_size + 1:i + 1]

        X = window[feature_columns].values
        y = window.iloc[-1]['co2flux']

        #features.append(X)
        #targets.append(y)

        sample_info = {
            'nav_lat': lat,
            'nav_lon': lon,
            'time_counter': window.iloc[-1]['time_counter']
        }
        #meta.append(sample_info)
        sample = Sample(X, y, sample_info)
        all_samples.append(sample)


# Save processed  data
output_dir = "/data/stu231428/Transformed_data_LSTM"
print("Saving  features, targets, and metadata...")

with open(f"{output_dir}/samples_train_global_0.005.pkl", "wb") as f:
    pickle.dump(all_samples, f)

# with open(f"{output_dir}/targets_train_global_0.005.pkl", "wb") as f:
#     pickle.dump(targets, f)

# with open(f"{output_dir}/meta_train_global_0.005.pkl", "wb") as f:
#     pickle.dump(meta, f)


