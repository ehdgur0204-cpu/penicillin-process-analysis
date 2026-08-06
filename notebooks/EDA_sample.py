import pandas as pd
from pathlib import Path

# Load the sample CSV file
csv_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "100_Batches_IndPenSim_V3_sample_1000.csv"
df = pd.read_csv(csv_path)

# Verify the data was loaded correctly
print("Loaded data shape:", df.shape)
print(df.head())
