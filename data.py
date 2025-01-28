from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()

# Convert to a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add the target column
df['target'] = data.target

# Save the dataset to a CSV file
df.to_csv('breast_cancer_data.csv', index=False)

print("Dataset saved as 'breast_cancer_data.csv'")