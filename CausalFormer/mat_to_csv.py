import scipy.io
import pandas as pd

# Load the .mat file
mat_file_path = '/home/tobias/Downloads/sims/sim1.mat'  # Replace with your actual file path
mat_data = scipy.io.loadmat(mat_file_path)

# Extract the 'ts' variable (the time series data)
ts_data = mat_data['ts']

# Convert the numpy array to a pandas DataFrame
# Assuming each column represents a time series and each row represents a time point
df = pd.DataFrame(ts_data)

# Optionally, add column headers if needed (for example, Time Series 1, Time Series 2, etc.)
df.columns = [f'Time Series {i+1}' for i in range(df.shape[1])]

# Save the DataFrame to a CSV file
csv_file_path = 'output_time_series.csv'  # Replace with the desired output file path
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved to: {csv_file_path}")
