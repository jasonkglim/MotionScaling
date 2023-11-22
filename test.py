import numpy as np
import math
import pandas as pd


# Create a DataFrame with a 'clutch' column
data = {'clutch': [True, True, False, False, True, True, False, False, True, True]}
df = pd.DataFrame(data)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Calculate the number of transitions from True to False using the provided code
transitions = sum((df['clutch']) & (df['clutch'].shift(-1) == False))

# Print the result
print("\nNumber of transitions from True to False:", transitions)
