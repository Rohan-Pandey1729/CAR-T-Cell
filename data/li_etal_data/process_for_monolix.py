import pandas as pd
import numpy as np

df_2b = pd.read_csv("2B.csv")
df_3a = pd.read_csv("3A.csv")

# Process 2B data
# - Convert y values using log10(10^8 * y) based on estimate of 10^10 lymphocytes in body
# - Set negative y values to 0
# - Apply max(0, x) to final observations
df_2b["observation"] = df_2b["y"].apply(lambda x: 0 if x < 0 else np.log10(10**8 * x))
df_2b["observation"] = df_2b["observation"].apply(lambda x: max(0, x))
df_2b["observation_id"] = 0
df_2b["observation_type"] = "T"
df_2b = df_2b.rename(columns={"day": "time", "patient": "id"})

# Process 3A data
# - Exclude rows where time <= 5 (these might be before T-cell infusion is complete)
# - Convert y values using log10(7 * 10^7 * y) based on estimate of 7 * 10^9 CD3+ lymphocytes in body
# - Set negative y values to 0
# - Apply max(0, x) to final observations
df_3a = df_3a[df_3a["day"] > 5]  # Filter out rows where time <= 5
df_3a["observation"] = df_3a["y"].apply(
    lambda x: 0 if x < 0 else np.log10(7 * 10**7 * x)
)
df_3a["observation"] = df_3a["observation"].apply(lambda x: max(0, x))
df_3a["observation_id"] = 1
df_3a["observation_type"] = "C"
df_3a = df_3a.rename(columns={"day": "time", "patient": "id"})

# Combine the dataframes
combined_df = pd.concat([df_2b, df_3a], ignore_index=True)

# Select and order the required columns
result_df = combined_df[
    ["time", "observation", "id", "observation_id", "observation_type"]
]

# Save to CSV
result_df.to_csv("data_for_monolix.csv", index=False)
