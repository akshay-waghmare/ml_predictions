import pandas as pd
from io import StringIO

# Prompt user to paste data
print("Paste your data here (end with an empty line):")
data_lines = []
while True:
    line = input()
    if line.strip() == '':
        break
    data_lines.append(line)

data = '\n'.join(data_lines)

# Convert to DataFrame
df = pd.read_csv(StringIO(data), sep='\t')

# Replace '-' with '0'
df.replace('-', '0', inplace=True)

# Remove '*' from HS column
df['HS'] = df['HS'].str.replace('*', '', regex=False)

# Drop the last column
df = df.iloc[:, :-1]

# Ask user for output filename
output_filename = input("Enter the output filename (with .csv extension): ")

# Save to CSV
df.to_csv(output_filename, index=False)

print(f"Data saved to {output_filename}")